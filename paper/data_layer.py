"""Real market-data layer for the QNLPAI experiments.

Provenance-first and reproducible. `load_universe()` returns annualized (mu, Sigma) for a real asset
universe with a recorded `source`:

  * "wrds-crsp"  — gold-standard CRSP daily returns via WRDS (needs the user's WRDS credentials; see
                   `wrds_crsp_returns`). This is the production/camera-ready source.
  * "snapshot"   — a fixed CSV snapshot of real adjusted closes (paper/data/universe_prices.csv), pulled
                   once from a public source. Fully reproducible offline; what the released figures use.

Covariance uses Ledoit--Wolf shrinkage (well-conditioned, standard in empirical finance).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
SNAPSHOT = os.path.join(HERE, "data", "universe_prices.csv")
TRADING_DAYS = 252


def _annualize(returns: pd.DataFrame, shrink: bool = True):
    mu = returns.mean().values * TRADING_DAYS
    if shrink:
        try:
            from sklearn.covariance import LedoitWolf
            cov = LedoitWolf().fit(returns.values).covariance_ * TRADING_DAYS
        except Exception:  # noqa: BLE001
            cov = np.cov(returns.values, rowvar=False) * TRADING_DAYS
    else:
        cov = np.cov(returns.values, rowvar=False) * TRADING_DAYS
    return mu, cov


def load_snapshot():
    """Real adjusted closes captured once to CSV -> annualized (mu, Sigma). Deterministic."""
    px = pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True).dropna(how="all").ffill().dropna()
    rets = px.pct_change().dropna()
    mu, sigma = _annualize(rets)
    return {"tickers": list(px.columns), "mu": mu, "sigma": sigma, "returns": rets,
            "source": "snapshot", "start": str(px.index.min())[:10], "end": str(px.index.max())[:10],
            "n_days": int(px.shape[0])}


def wrds_crsp_returns(tickers, start="2022-01-01", end="2024-12-31", wrds_username=None):
    """Gold-standard path: CRSP daily returns via WRDS. The USER runs this (it needs WRDS credentials,
    which an assistant must never enter). On first use `wrds.Connection()` prompts once and writes a
    ~/.pgpass; thereafter it is non-interactive. Maps tickers -> PERMNO via crsp.stocknames, then pulls
    crsp.dsf. Returns the same dict shape as load_snapshot() with source='wrds-crsp'."""
    import wrds  # noqa: F401 — only needed on this path

    db = wrds.Connection(wrds_username=wrds_username) if wrds_username else wrds.Connection()
    try:
        tk = ",".join(f"'{t.upper()}'" for t in tickers)
        names = db.raw_sql(f"select distinct permno, ticker from crsp.stocknames where ticker in ({tk})")
        permno_map = {r.ticker.upper(): int(r.permno) for r in names.itertuples()}
        permnos = ",".join(str(p) for p in permno_map.values())
        dsf = db.raw_sql(
            f"select permno, date, ret from crsp.dsf "
            f"where permno in ({permnos}) and date between '{start}' and '{end}'",
            date_cols=["date"])
        inv = {v: k for k, v in permno_map.items()}
        dsf["ticker"] = dsf["permno"].map(inv)
        wide = dsf.pivot(index="date", columns="ticker", values="ret").dropna()
        wide = wide[[t for t in tickers if t in wide.columns]]
        mu, sigma = _annualize(wide)
        return {"tickers": list(wide.columns), "mu": mu, "sigma": sigma, "returns": wide,
                "source": "wrds-crsp", "start": start, "end": end, "n_days": int(wide.shape[0])}
    finally:
        db.close()


def load_universe(source: str = "auto", wrds_username: str | None = None):
    """source='auto' uses WRDS if credentials are clearly available, else the reproducible snapshot."""
    if source == "wrds" or (source == "auto" and (wrds_username or os.path.exists(
            os.path.expanduser("~/.pgpass")))):
        try:
            return wrds_crsp_returns(["AAPL", "MSFT", "NVDA", "XOM", "JPM", "GLD"],
                                     wrds_username=wrds_username)
        except Exception as e:  # noqa: BLE001 — fall back to the snapshot, but record why
            snap = load_snapshot(); snap["wrds_error"] = f"{type(e).__name__}: {str(e)[:80]}"; return snap
    return load_snapshot()


if __name__ == "__main__":
    u = load_universe()
    print(f"source={u['source']}  tickers={u['tickers']}  {u['start']}..{u['end']}  ({u['n_days']}d)")
    print("ann mu %:", dict(zip(u["tickers"], (u["mu"] * 100).round(1))))
