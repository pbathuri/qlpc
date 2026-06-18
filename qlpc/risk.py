"""Stochastic scenarios + classical risk metrics (VaR / CVaR).

This is the deterministic oracle the quantum amplitude-estimation engine is validated against: the same
loss distribution, measured classically by Monte-Carlo. The quantum engine (quantum_risk.py) targets the
SAME quantities with a quadratic sample-complexity speedup.
"""
from __future__ import annotations

import numpy as np


def simulate_scenarios(mu: np.ndarray, sigma: np.ndarray, n_scenarios: int = 20000,
                       horizon_days: int = 21, seed: int = 11) -> np.ndarray:
    """Multivariate-normal scenario returns over `horizon_days` (per-asset), shape (S, n).
    A correlated-Gaussian stand-in for richer processes (Heston/GARCH) — swap-in ready."""
    rng = np.random.default_rng(seed)
    n = len(mu)
    daily_mu = mu / 252.0
    daily_cov = sigma / 252.0
    # sum of horizon daily draws ~ N(h*mu, h*cov)
    mean = daily_mu * horizon_days
    cov = daily_cov * horizon_days
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(n))
    z = rng.standard_normal((n_scenarios, n))
    return mean + z @ L.T


def portfolio_losses(weights: np.ndarray, scenarios: np.ndarray) -> np.ndarray:
    """Loss (= -return) of the portfolio under each scenario, shape (S,)."""
    return -(scenarios @ weights)


def var_cvar(losses: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """Value-at-Risk and Conditional VaR (expected shortfall) at level alpha, classical MC estimate."""
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var
    return var, cvar
