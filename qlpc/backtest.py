"""Dynamic rebalancing backtest — the "dynamic" in dynamic portfolio construction.

Walks the history forward: every `rebalance_every` days, re-construct the portfolio on the trailing
window (QLPC pipeline), hold, and book realized P&L net of turnover cost. Reports the realized track
record (return / vol / Sharpe / max-drawdown / turnover).

This is a clean vectorized stand-in for execution; the PRODUCTION execution + event-driven backtest
(realistic fills, slippage, fees, broker adapters) is `nautechsystems/nautilus_trader` — `to_orders()`
below is the handoff: target weights -> rebalancing orders nautilus routes.
"""
from __future__ import annotations

import numpy as np

from .data import MarketData, get_market_data
from .pipeline import construct_portfolio


def to_orders(target_weights: dict[str, float], nav: float, prices: dict[str, float]) -> list[dict]:
    """Target weights -> notional rebalancing orders (the nautilus_trader execution handoff)."""
    return [{"symbol": s, "side": "REBALANCE", "target_notional": round(nav * w, 2),
             "target_qty": round(nav * w / prices.get(s, 1.0), 4)}
            for s, w in target_weights.items() if w > 1e-6]


def backtest(tickers: list[str], *, allow_live: bool = False, lookback: int = 252,
             rebalance_every: int = 21, brief: str = "balanced, 4 names", cost_bps: float = 10.0,
             quantum: bool = False) -> dict:
    """Rolling-rebalance backtest of the QLPC construction. `quantum=False` keeps the many-window loop
    fast (classical path); set True to run QAOA/IQAE at each rebalance."""
    md = get_market_data(tickers, allow_live=allow_live)
    R, names = md.returns, md.tickers
    T, n = R.shape
    equity, turnovers = [1.0], []
    prev_w = np.zeros(n)
    t = lookback
    n_rebal = 0
    while t + rebalance_every <= T:
        win = R[t - lookback:t]
        window_md = MarketData(names, win.mean(0) * 252, np.cov(win, rowvar=False) * 252, win, md.source)
        res = construct_portfolio(brief, md=window_md, quantum=quantum, ground=False)
        w = np.zeros(n)
        for tk, wt in res["weights"].items():
            w[names.index(tk)] = wt
        to = float(np.abs(w - prev_w).sum())
        turnovers.append(to)
        eq = equity[-1] * (1 - cost_bps / 1e4 * to)          # pay turnover cost at rebalance
        for r in R[t:t + rebalance_every] @ w:               # hold the book
            eq *= (1 + r)
            equity.append(eq)
        prev_w, t, n_rebal = w, t + rebalance_every, n_rebal + 1

    eq = np.array(equity)
    rets = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])
    periods = max(1, len(rets))
    ann_ret = float(eq[-1] ** (252 / periods) - 1)
    ann_vol = float(rets.std() * np.sqrt(252))
    dd = 1 - eq / np.maximum.accumulate(eq)
    return {
        "tickers": tickers, "data_source": md.source, "n_rebalances": n_rebal,
        "final_equity": round(float(eq[-1]), 4),
        "annualized_return": round(ann_ret, 4),
        "annualized_vol": round(ann_vol, 4),
        "sharpe": round(ann_ret / ann_vol, 3) if ann_vol > 0 else 0.0,
        "max_drawdown": round(float(dd.max()), 4),
        "avg_turnover": round(float(np.mean(turnovers)) if turnovers else 0.0, 4),
        "quantum_per_rebalance": quantum,
        "equity_curve": [round(float(x), 4) for x in eq[:: max(1, len(eq) // 30)]],
    }
