"""QLPC orchestration — the full Quantum-LLM Personalized Constructor flow.

  brief (NL) ──► graph.ground (personalize) ──► llm.elicit_constraints
                                                        │
        data ──► quantum_select (QAOA: which K assets) ─┤
                                                        ▼
            simulate scenarios ──► convex.mean_cvar (exact weights on the subset)
                                                        │
                          quantum_risk (IQAE: tail CVaR) ┤
                                                        ▼
                            llm.explain (grounded rationale) ──► PortfolioResult
"""
from __future__ import annotations

import numpy as np

from . import convex, graph, llm
from .data import get_market_data
from .quantum_risk import quantum_cvar
from .quantum_select import quantum_select
from .risk import portfolio_losses, simulate_scenarios

DEFAULT_UNIVERSE = ["AAPL", "MSFT", "NVDA", "JPM", "JNJ", "GLD"]


def construct_portfolio(brief: str, tickers: list[str] | None = None, *,
                        allow_live: bool = False, n_qubits: int = 6, risk_factor: float = 0.5,
                        select_reps: int = 1, select_maxiter: int = 80, quantum: bool = True,
                        md=None, ground: bool = True) -> dict:
    tickers = tickers or DEFAULT_UNIVERSE
    graph_ctx = graph.ground(f"portfolio risk preferences {brief}") if ground else {"context": [], "source": "off"}
    cons = llm.elicit_constraints(brief)

    md = md or get_market_data(tickers, allow_live=allow_live)
    budget = int(np.clip(cons["budget"], 2, md.n - 1))

    selection = quantum_select(md.mu, md.sigma, budget=budget, risk_factor=risk_factor,
                               reps=select_reps, maxiter=select_maxiter, quantum=quantum)
    idx = selection["selected"]
    sub_tickers = [md.tickers[i] for i in idx]
    sub_mu, sub_sig = md.mu[idx], md.sigma[np.ix_(idx, idx)]

    scenarios = simulate_scenarios(sub_mu, sub_sig, n_scenarios=12000)
    w = convex.mean_cvar(sub_mu, scenarios, alpha=cons["alpha"], lam=cons["lam"])
    losses = portfolio_losses(w, scenarios)
    risk = quantum_cvar(losses, alpha=cons["alpha"], n_qubits=n_qubits, quantum=quantum)

    weights = {sub_tickers[i]: round(float(w[i]), 4) for i in range(len(idx))}
    result = {
        "brief": brief,
        "constraints": cons,
        "data_source": md.source,
        "universe": tickers,
        "selection": {"method": selection["method"], "tickers": sub_tickers,
                      "optimal_vs_classical": selection.get("optimal")},
        "weights": weights,
        "expected_return": round(float(sub_mu @ w), 4),
        "risk": risk,
        "graph": {"source": graph_ctx["source"], "context_used": len(graph_ctx["context"])},
    }
    result["explanation"] = llm.explain(result, graph_ctx["context"])
    return result
