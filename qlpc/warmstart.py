"""QAOA warm-start — a performance-enhancing model for the quantum selection layer.

Cold QAOA starts from the uniform superposition |+>^n and can need many iterations / get stuck. A
warm-start biases the initial state toward a good CLASSICAL relaxation, so QAOA starts near a strong
solution and converges faster to higher quality (Egger, Marecek, Woerner 2021, "Warm-starting quantum
optimization"; cf. your AutoResearchClaw Q06 NN-warm-start-for-QAOA topic).

We solve the continuous mean-variance relaxation, map each asset's relaxed weight to an inclusion
probability c_i, and prepare |psi0> = ⊗ Ry(2·arcsin√c_i)|0> as QAOA's initial state. Returns None on any
issue so the caller falls back to a cold start.
"""
from __future__ import annotations

import numpy as np


def warm_start_state(mu: np.ndarray, sigma: np.ndarray, risk_factor: float):
    """A QuantumCircuit preparing the relaxation-biased QAOA initial state, or None."""
    try:
        from qiskit import QuantumCircuit

        from .convex import mean_variance
        n = len(mu)
        w = mean_variance(mu, sigma, risk_aversion=max(0.5, risk_factor * 10))   # continuous relaxation
        c = w / (w.max() + 1e-12)
        c = np.clip(c, 0.05, 0.95)                          # keep angles away from poles
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.ry(2 * float(np.arcsin(np.sqrt(c[i]))), i)
        return qc
    except Exception:
        return None
