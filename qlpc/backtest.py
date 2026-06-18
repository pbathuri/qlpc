"""Dynamic rebalancing backtest — generic, joblib-parallel, strategy-agnostic.

`run_strategy` walks history forward, re-computing weights every `rebalance_every` days on the trailing
window via any `weight_fn(window_returns, tickers) -> {ticker: weight}`. The independent per-window weight
computations are evaluated in PARALLEL (joblib); P&L is then compounded sequentially net of turnover cost.
This is the engine the benchmark harness uses to compare strategies head-to-head on identical data.

`to_orders` is the handoff to nautechsystems/nautilus_trader — the production event-driven execution +
backtest engine (realistic fills/slippage/fees, broker adapters). QLPC constructs; nautilus executes.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .data import MarketData, get_market_data


def to_orders(target_weights: dict[str, float], nav: float, prices: dict[str, float]) -> list[dict]:
    """Target weights -> notional rebalancing orders (the nautilus_trader execution handoff)."""
    return [{"symbol": s, "side": "REBALANCE", "target_notional": round(nav * w, 2),
             "target_qty": round(nav * w / prices.get(s, 1.0), 4)}
            for s, w in target_weights.items() if w > 1e-6]


def _metrics(equity: np.ndarray, turnovers: list[float]) -> dict:
    rets = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0.0])
    periods = max(1, len(rets))
    ann_ret = float(equity[-1] ** (252 / periods) - 1)
    ann_vol = float(rets.std() * np.sqrt(252))
    dd = 1 - equity / np.maximum.accumulate(equity)
    return {"final_equity": round(float(equity[-1]), 4),
            "annualized_return": round(ann_ret, 4), "annualized_vol": round(ann_vol, 4),
            "sharpe": round(ann_ret / ann_vol, 3) if ann_vol > 0 else 0.0,
            "max_drawdown": round(float(dd.max()), 4),
            "avg_turnover": round(float(np.mean(turnovers)) if turnovers else 0.0, 4)}


def run_strategy(md: MarketData, weight_fn: Callable, *, lookback: int = 252, rebalance_every: int = 21,
                 cost_bps: float = 10.0, n_jobs: int = 1) -> dict:
    """Rolling backtest of `weight_fn` on `md`. Per-window weights computed in parallel (n_jobs)."""
    R, names = md.returns, md.tickers
    T, n = R.shape
    starts = list(range(lookback, T - rebalance_every + 1, rebalance_every))

    def _w(t):
        return weight_fn(R[t - lookback:t], names)

    if n_jobs != 1 and len(starts) > 1:
        try:
            from joblib import Parallel, delayed
            wlist = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_w)(t) for t in starts)
        except Exception:
            wlist = [_w(t) for t in starts]
    else:
        wlist = [_w(t) for t in starts]

    equity, turnovers, prev = [1.0], [], np.zeros(n)
    for t, wd in zip(starts, wlist):
        w = np.zeros(n)
        for tk, wt in (wd or {}).items():
            if tk in names:
                w[names.index(tk)] = wt
        to = float(np.abs(w - prev).sum())
        turnovers.append(to)
        eq = equity[-1] * (1 - cost_bps / 1e4 * to)
        for r in R[t:t + rebalance_every] @ w:
            eq *= (1 + r)
            equity.append(eq)
        prev = w
    m = _metrics(np.array(equity), turnovers)
    m.update(n_rebalances=len(starts), data_source=md.source)
    return m


def _qlpc_weight_fn(brief: str, quantum: bool, warm_start: bool):
    """A weight_fn that runs the QLPC construction on a window."""
    from .pipeline import construct_portfolio

    def fn(window_returns, names):
        wmd = MarketData(names, window_returns.mean(0) * 252,
                         np.cov(window_returns, rowvar=False) * 252, window_returns, "window")
        res = construct_portfolio(brief, md=wmd, quantum=quantum, ground=False,
                                  select_reps=1, select_maxiter=60)
        # construct_portfolio may set warm_start via select; pass through pipeline if supported
        return res["weights"]
    return fn


def backtest(tickers: list[str], *, allow_live: bool = False, lookback: int = 252,
             rebalance_every: int = 21, brief: str = "balanced, 4 names", cost_bps: float = 10.0,
             quantum: bool = False, n_jobs: int = 1) -> dict:
    """Convenience: rolling-rebalance backtest of the QLPC construction."""
    md = get_market_data(tickers, allow_live=allow_live)
    out = run_strategy(md, _qlpc_weight_fn(brief, quantum, False),
                       lookback=lookback, rebalance_every=rebalance_every, cost_bps=cost_bps, n_jobs=n_jobs)
    out.update(tickers=tickers, quantum_per_rebalance=quantum)
    return out
