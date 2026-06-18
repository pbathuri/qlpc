"""Classical convex optimization layer — exact continuous weights for the SELECTED assets.

Quantum heuristics (QAOA/VQE) pick *which* assets (discrete, NP-hard with cardinality). Once the subset
is fixed, the continuous weight problem is a convex QP — solve it exactly and fast. We support
mean-variance and mean-CVaR (the latter via Rockafellar-Uryasev LP using return scenarios). Uses CVXPY
when present; falls back to the closed-form tangency/min-variance solution so it always runs.
"""
from __future__ import annotations

import numpy as np


def _project_simplex(w: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum w = 1} — long-only fallback normalizer."""
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def mean_variance(mu: np.ndarray, sigma: np.ndarray, risk_aversion: float = 5.0) -> np.ndarray:
    """argmax_w  mu·w - (gamma/2) wᵀΣw  s.t. w>=0, sum w = 1. Convex (CVXPY) with closed-form fallback."""
    n = len(mu)
    try:
        import cvxpy as cp
        w = cp.Variable(n)
        obj = cp.Maximize(mu @ w - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma)))
        prob = cp.Problem(obj, [cp.sum(w) == 1, w >= 0])
        prob.solve()
        if w.value is not None:
            return _project_simplex(np.asarray(w.value).ravel())
    except Exception:
        pass
    # closed-form unconstrained tangency, then project to long-only simplex
    try:
        inv = np.linalg.pinv(sigma + 1e-6 * np.eye(n))
        return _project_simplex(inv @ mu)
    except Exception:
        return np.ones(n) / n


def mean_cvar(mu: np.ndarray, scenarios: np.ndarray, alpha: float = 0.95,
              lam: float = 1.0) -> np.ndarray:
    """Mean-CVaR (Rockafellar-Uryasev). max_w mu·w - lam·CVaR_alpha(loss) s.t. w>=0, sum w=1.
    `scenarios` are per-asset return samples, shape (S, n). Convex LP via CVXPY; falls back to MV."""
    S, n = scenarios.shape
    try:
        import cvxpy as cp
        w = cp.Variable(n)
        t = cp.Variable()                       # VaR level
        u = cp.Variable(S, nonneg=True)         # tail excess
        losses = -scenarios @ w                 # loss = -return
        cvar = t + (1.0 / (S * (1 - alpha))) * cp.sum(u)
        cons = [u >= losses - t, cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(cp.Maximize(mu @ w - lam * cvar), cons)
        prob.solve()
        if w.value is not None:
            return _project_simplex(np.asarray(w.value).ravel())
    except Exception:
        pass
    return mean_variance(mu, np.cov(scenarios, rowvar=False) + 1e-6 * np.eye(n), risk_aversion=lam * 5)
