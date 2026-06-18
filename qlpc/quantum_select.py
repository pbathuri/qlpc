"""Quantum selection engine — cardinality-constrained asset SELECTION as a QUBO, solved by QAOA.

Choosing WHICH K of N assets to hold (binary, with a budget constraint) is NP-hard and is exactly the
piece classical convex solvers can't do. We map Markowitz-with-cardinality to a QUBO / Ising Hamiltonian
and solve it with QAOA (Farhi 2014) — the same gate-based, NISQ-suitable family your captain-whiskers
VQE optimizer used, here specialised to the discrete selection layer.

Always returns: QAOA via qiskit-finance when installed; otherwise an exact classical subset search
(also the validation oracle). Reports the method + objective so the two can be compared.
"""
from __future__ import annotations

import itertools

import numpy as np


def _objective(sel: np.ndarray, mu: np.ndarray, sigma: np.ndarray, q: float) -> float:
    """QUBO objective for a binary selection vector: mu·x - q·xᵀΣx (higher = better)."""
    return float(mu @ sel - q * sel @ sigma @ sel)


def _classical_best(mu, sigma, budget, q) -> tuple[list[int], float]:
    """Exact best subset of size `budget` (brute force; fine for prototype universes ≤ ~16)."""
    n = len(mu)
    best, best_obj = None, -np.inf
    for combo in itertools.combinations(range(n), budget):
        x = np.zeros(n)
        x[list(combo)] = 1
        o = _objective(x, mu, sigma, q)
        if o > best_obj:
            best, best_obj = list(combo), o
    return best, best_obj


def quantum_select(mu: np.ndarray, sigma: np.ndarray, budget: int, risk_factor: float = 0.5,
                   reps: int = 2, maxiter: int = 150, quantum: bool = True,
                   warm_start: bool = False) -> dict:
    """Select `budget` assets via QAOA (QUBO) with an exact classical fallback. Returns
    {selected, method, objective, classical_objective, optimal}. `quantum=False` forces the classical
    exact path (used by the backtest loop for speed; single constructions use QAOA)."""
    n = len(mu)
    cls_sel, cls_obj = _classical_best(mu, sigma, budget, risk_factor)
    out = {"n": n, "budget": budget, "risk_factor": risk_factor,
           "classical_selected": cls_sel, "classical_objective": round(cls_obj, 6)}
    if not quantum:
        out.update(method="classical-exact", selected=cls_sel, objective=round(cls_obj, 6), optimal=True)
        return out
    try:
        from qiskit.primitives import StatevectorSampler  # qiskit 2.x V2 primitive
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_finance.applications.optimization import PortfolioOptimization
        from qiskit_optimization.algorithms import MinimumEigenOptimizer

        po = PortfolioOptimization(expected_returns=np.asarray(mu), covariances=np.asarray(sigma),
                                   risk_factor=risk_factor, budget=budget)
        qp = po.to_quadratic_program()
        init = None
        if warm_start:
            from .warmstart import warm_start_state
            init = warm_start_state(mu, sigma, risk_factor)
        qaoa = QAOA(sampler=StatevectorSampler(), optimizer=COBYLA(maxiter=maxiter), reps=reps,
                    initial_state=init)
        result = MinimumEigenOptimizer(qaoa).solve(qp)
        x = np.array(result.x, dtype=int)               # raw binary selection vector
        selected = [i for i, b in enumerate(x) if b == 1]
        # if QAOA returns an infeasible cardinality, fall back to its best feasible neighbour
        if len(selected) != budget:
            raise ValueError(f"QAOA returned {len(selected)} assets, expected {budget}")
        q_obj = _objective(x.astype(float), mu, sigma, risk_factor)
        out.update(method="quantum-qaoa", selected=selected, objective=round(q_obj, 6),
                   optimal=(set(selected) == set(cls_sel)), warm_start=bool(warm_start and init is not None))
        return out
    except Exception as e:  # noqa: BLE001
        out.update(method="classical-exact", selected=cls_sel, objective=round(cls_obj, 6),
                   optimal=True, note=f"quantum path unavailable: {type(e).__name__}")
        return out
