"""Market data layer. Returns expected-return vector mu and covariance Sigma for a universe of assets.

Tries live data (yfinance); falls back to a deterministic synthetic market (seeded) so the whole
pipeline ALWAYS runs offline / in CI. Honest about which source was used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRADING_DAYS = 252


@dataclass
class MarketData:
    tickers: list[str]
    mu: np.ndarray            # annualized expected returns, shape (n,)
    sigma: np.ndarray         # annualized covariance, shape (n, n)
    returns: np.ndarray       # daily returns, shape (T, n)
    source: str               # "yfinance" | "synthetic"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.tickers)


def _synthetic(tickers: list[str], days: int = 756, seed: int = 7) -> MarketData:
    """Deterministic synthetic market: correlated GBM daily returns with a couple of factors.
    Distinct per-ticker drift/vol so selection + risk have real structure to find."""
    rng = np.random.default_rng(seed)
    n = len(tickers)
    # two latent factors + idiosyncratic noise -> a realistic, non-trivial covariance
    loadings = rng.normal(0, 1, size=(n, 2))
    f = rng.normal(0, 0.012, size=(days, 2))
    idio = rng.normal(0, 1, size=(n,)) * 0.008 + 0.004
    daily = f @ loadings.T + rng.normal(0, 1, size=(days, n)) * idio
    drift = (np.linspace(0.02, 0.16, n) + rng.normal(0, 0.02, n)) / TRADING_DAYS
    daily = daily + drift
    mu = daily.mean(0) * TRADING_DAYS
    sigma = np.cov(daily, rowvar=False) * TRADING_DAYS
    return MarketData(list(tickers), mu, sigma, daily, "synthetic",
                      {"days": days, "seed": seed})


def get_market_data(tickers: list[str], period: str = "3y", *, allow_live: bool = True) -> MarketData:
    """Expected returns + covariance for `tickers`. Live via yfinance if available, else synthetic."""
    if allow_live:
        try:
            import yfinance as yf  # noqa
            px = yf.download(tickers, period=period, progress=False, auto_adjust=True)["Close"]
            px = px.dropna(how="all").dropna(axis=1)
            rets = px.pct_change().dropna()
            if rets.shape[0] > 60 and rets.shape[1] >= 2:
                cols = list(rets.columns)
                daily = rets.to_numpy()
                return MarketData(cols, daily.mean(0) * TRADING_DAYS,
                                  np.cov(daily, rowvar=False) * TRADING_DAYS, daily, "yfinance",
                                  {"period": period, "obs": int(rets.shape[0])})
        except Exception:
            pass
    return _synthetic(tickers)
