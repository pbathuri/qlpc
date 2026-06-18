"""QLPC tests — portable: pass whether or not qiskit is installed (quantum modules fall back to
classical, and we assert on RESULT validity + the quantum/classical agreement, not the method)."""
import numpy as np

from qlpc import convex, llm
from qlpc.data import get_market_data
from qlpc.pipeline import construct_portfolio
from qlpc.quantum_risk import quantum_cvar
from qlpc.quantum_select import quantum_select
from qlpc.risk import portfolio_losses, simulate_scenarios, var_cvar


def _md():
    return get_market_data(["A", "B", "C", "D", "E", "F"], allow_live=False)


def test_data_synthetic_shapes():
    md = _md()
    assert md.n == 6 and md.mu.shape == (6,) and md.sigma.shape == (6, 6) and md.source == "synthetic"


def test_convex_weights_are_valid_simplex():
    md = _md()
    w = convex.mean_variance(md.mu, md.sigma, risk_aversion=5)
    assert abs(w.sum() - 1) < 1e-6 and (w >= -1e-9).all()


def test_mean_cvar_runs_and_normalizes():
    md = _md()
    sc = simulate_scenarios(md.mu, md.sigma, n_scenarios=3000)
    w = convex.mean_cvar(md.mu, sc, alpha=0.95, lam=1.0)
    assert abs(w.sum() - 1) < 1e-6 and (w >= -1e-9).all()


def test_var_cvar_ordering():
    losses = np.random.default_rng(0).normal(0, 1, 50000)
    var, cvar = var_cvar(losses, 0.95)
    assert cvar >= var                                  # expected shortfall >= VaR


def test_quantum_select_returns_budget_and_is_optimal():
    md = _md()
    r = quantum_select(md.mu, md.sigma, budget=3, risk_factor=0.5, reps=1, maxiter=40)
    assert len(r["selected"]) == 3
    assert r["optimal"] is True                         # quantum OR classical-exact must hit the optimum
    assert r["method"] in {"quantum-qaoa", "classical-exact"}


def test_quantum_cvar_matches_classical():
    md = _md()
    sc = simulate_scenarios(md.mu, md.sigma, n_scenarios=6000)
    w = np.ones(md.n) / md.n
    r = quantum_cvar(portfolio_losses(w, sc), alpha=0.95, n_qubits=6)
    assert r["cvar"] > 0
    # quantum (discretized) within 20% of the classical MC oracle; classical path is exact
    assert abs(r["cvar"] - r["classical_cvar"]) / r["classical_cvar"] < 0.20


def test_llm_constraint_parser():
    c = llm.elicit_constraints("conservative low-drawdown, about 4 names, tail-aware")
    assert c["budget"] == 4 and c["risk_aversion"] >= 5 and c["alpha"] == 0.99


def test_end_to_end_pipeline():
    r = construct_portfolio("balanced, 3 names", tickers=["A", "B", "C", "D", "E"], allow_live=False)
    assert len(r["weights"]) == 3
    assert abs(sum(r["weights"].values()) - 1) < 1e-6
    assert r["risk"]["cvar"] > 0 and r["explanation"]
