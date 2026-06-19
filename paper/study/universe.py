"""Objective asset universe + reproducible price snapshot.

To avoid universe-selection bias, the pool is a published index (Dow Jones Industrial Average, 30
constituents) rather than a hand-picked set. Each QAOA instance later draws a SUBSET of this pool, so no
single favorable instance can drive the conclusions. The production path is WRDS/CRSP (see
`paper.data_layer.wrds_crsp_returns`); this module provides a fixed, offline-reproducible snapshot of real
adjusted closes for development and for anyone without WRDS access.
"""
from __future__ import annotations

import io
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
SNAP = os.path.join(HERE, "data")
os.makedirs(SNAP, exist_ok=True)
TRADING_DAYS = 252

# Dow Jones Industrial Average constituents (objective index membership, as of 2025-06). Documented so
# the universe is reproducible and not author-selected. (CRSP/PERMNO mapping handled in data_layer.)
DOW30 = ["AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
         "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
         "MSFT", "NKE", "NVDA", "PG", "TRV", "UNH", "V", "VZ", "WMT", "DOW"]


def _stooq(sym: str) -> pd.Series | None:
    try:
        raw = urllib.request.urlopen(f"https://stooq.com/q/d/l/?s={sym.lower()}.us&i=d", timeout=30).read().decode()
        d = pd.read_csv(io.StringIO(raw), parse_dates=["Date"]).set_index("Date")["Close"]
        return d if len(d) > 50 else None
    except Exception:  # noqa: BLE001
        return None


def fetch_snapshot(tickers=DOW30, period_days=756, path=None):
    """Pull real daily closes once and cache to CSV (reproducible thereafter). yfinance, stooq fallback."""
    path = path or os.path.join(SNAP, "dow30_prices.csv")
    prices = None
    try:
        import tempfile
        import yfinance as yf
        cache = os.path.join(tempfile.gettempdir(), "yf_fresh_cache"); os.makedirs(cache, exist_ok=True)
        try:
            yf.set_tz_cache_location(cache)
        except Exception:  # noqa: BLE001
            pass
        df = yf.download(tickers, period="3y", interval="1d", auto_adjust=True, progress=False, threads=False)
        px = df["Close"].dropna(how="all")
        if px.shape[0] > 100:
            prices = px
    except Exception:  # noqa: BLE001
        pass
    if prices is None or prices.shape[1] < len(tickers) * 0.8:
        cols = {t: s for t in tickers if (s := _stooq(t)) is not None}
        prices = pd.DataFrame(cols)
    prices = prices.dropna(how="all").ffill().tail(period_days)
    prices = prices.dropna(axis=1)                      # keep only fully-populated names
    prices.to_csv(path)
    return path, list(prices.columns), prices.shape


def load_prices(path=None) -> pd.DataFrame:
    path = path or os.path.join(SNAP, "dow30_prices.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True).dropna(axis=1)


def window_moments(prices: pd.DataFrame, idx_tickers, start, end, shrink=True):
    """Annualized (mu, Sigma) from a date window — used for strict train/test separation."""
    sub = prices.loc[start:end, idx_tickers]
    rets = sub.pct_change().dropna()
    mu = rets.mean().values * TRADING_DAYS
    if shrink:
        try:
            from sklearn.covariance import LedoitWolf
            cov = LedoitWolf().fit(rets.values).covariance_ * TRADING_DAYS
        except Exception:  # noqa: BLE001
            cov = np.cov(rets.values, rowvar=False) * TRADING_DAYS
    else:
        cov = np.cov(rets.values, rowvar=False) * TRADING_DAYS
    return mu, cov, rets


if __name__ == "__main__":
    p, cols, shape = fetch_snapshot()
    print(f"snapshot -> {p}  ({shape[0]} days x {shape[1]} names)")
    print("universe:", cols)
