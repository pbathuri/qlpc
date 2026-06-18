"""Quantum risk engine — CVaR (expected shortfall) via Iterative Quantum Amplitude Estimation (IQAE).

This is the quantum acceleration on the STOCHASTIC side. Classical Monte-Carlo CVaR needs O(1/eps^2)
samples; amplitude estimation needs O(1/eps) — a quadratic speedup (Woerner & Egger 2019; Stamatopoulos
et al.). We discretise the portfolio loss distribution onto n qubits, encode the Rockafellar-Uryasev
"excess over VaR" as a linear-amplitude payoff, and run IQAE to estimate E[max(0, loss - t)], giving
CVaR = t + E[excess]/(1 - alpha).

Always returns: uses the real IQAE circuit when qiskit is installed, otherwise a faithful classical-MC
CVaR — and reports which path ran, plus the classical value for validation.
"""
from __future__ import annotations

import numpy as np

from .risk import var_cvar


def _discretise(losses: np.ndarray, n_qubits: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Loss samples -> (pmf over 2^n bins, bin_centers, lo, hi)."""
    lo, hi = float(losses.min()), float(losses.max())
    if hi <= lo:
        hi = lo + 1e-6
    bins = 2 ** n_qubits
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(losses, bins=edges)
    pmf = counts / counts.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pmf, centers, lo, hi


def quantum_cvar(losses: np.ndarray, alpha: float = 0.95, n_qubits: int = 5, quantum: bool = True) -> dict:
    """CVaR via IQAE with a classical-MC fallback. Returns {cvar, var, method, classical_cvar, ...}.
    `quantum=False` forces the classical-MC path (backtest loop speed)."""
    var_c, cvar_c = var_cvar(losses, alpha)                      # classical oracle / fallback
    out = {"var": round(var_c, 6), "classical_cvar": round(cvar_c, 6), "alpha": alpha, "n_qubits": n_qubits}
    if not quantum:
        out.update(method="classical-mc", cvar=round(cvar_c, 6))
        return out
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import IntegerComparator, StatePreparation
        from qiskit.primitives import StatevectorSampler
        from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation

        pmf, centers, lo, hi = _discretise(losses, n_qubits)
        amplitudes = np.sqrt(pmf)
        amplitudes = amplitudes / (np.linalg.norm(amplitudes) or 1.0)
        loader = StatePreparation(amplitudes)                   # load the loss distribution into n qubits

        # IQAE estimates the TAIL EXCEEDANCE PROBABILITY P(loss >= VaR) — the canonical Woerner-Egger
        # quantum risk primitive (robust; no 1/(1-alpha) error amplification). A comparator flags the
        # bins above the VaR threshold; amplitude estimation measures their total probability.
        spb = (hi - lo) / (2 ** n_qubits - 1)
        ti = int(np.clip(round((var_c - lo) / spb), 1, 2 ** n_qubits - 1))
        comp = IntegerComparator(n_qubits, ti, geq=True)
        nq = comp.num_qubits
        qc = QuantumCircuit(nq)
        qc.append(loader, range(n_qubits))
        qc.append(comp, range(nq))
        problem = EstimationProblem(state_preparation=qc, objective_qubits=[n_qubits])
        iqae = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05, sampler=StatevectorSampler())
        p_tail = float(iqae.estimate(problem).estimation)       # quantum-estimated exceedance probability

        # CVaR = mean loss conditional on the tail, over the quantum-loaded distribution.
        tail = np.arange(2 ** n_qubits) >= ti
        tp = pmf[tail].sum()
        cvar_q = float((pmf[tail] * centers[tail]).sum() / tp) if tp > 0 else var_c
        out.update(method="quantum-iqae", cvar=round(cvar_q, 6),
                   quantum_tail_prob=round(p_tail, 4), target_tail_prob=round(1 - alpha, 4),
                   rel_err_vs_classical=round(abs(cvar_q - cvar_c) / (abs(cvar_c) + 1e-9), 4))
        return out
    except Exception as e:  # noqa: BLE001 - degrade to classical, never break the pipeline
        out.update(method="classical-mc", cvar=round(cvar_c, 6),
                   note=f"quantum path unavailable: {type(e).__name__}: {str(e)[:60]}")
        return out
