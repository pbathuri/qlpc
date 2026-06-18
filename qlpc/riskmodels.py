"""Enhanced risk + optimization models, ingraining two battle-tested, widely-cited frameworks:

  * scikit-learn's **Ledoit-Wolf shrinkage** covariance — a proven estimator that shrinks the noisy
    sample covariance toward a structured target, cutting out-of-sample estimation error (the #1 reason
    naive Markowitz under-performs). This is a real performance enhancer for every downstream optimizer.
  * **Riskfolio-Lib** — a highly-regarded portfolio-optimization library (CVaR, CDaR, EVaR, risk parity,
    HRP, 13+ risk measures), used here as the convex weight engine with proven solvers.

Both degrade to QLPC's in-house CVXPY layer if absent, so the pipeline still runs.
"""
from __future__ import annotations

import numpy as np


def shrunk_covariance(returns: np.ndarray, annualize: int = 252) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance (annualized). Falls back to the sample covariance."""
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(returns).covariance_
        return cov * annualize
    except Exception:
        return np.cov(returns, rowvar=False) * annualize


def riskfolio_weights(returns: np.ndarray, tickers: list[str], *, rm: str = "CVaR",
                      obj: str = "Sharpe", alpha: float = 0.05) -> dict | None:
    """Convex weights from Riskfolio-Lib with Ledoit-Wolf mu/cov. `rm` risk measure (CVaR/CDaR/MV...),
    `obj` objective (Sharpe/MinRisk/Utility). Returns {ticker: weight} or None if unavailable."""
    try:
        import pandas as pd
        import riskfolio as rp
        df = pd.DataFrame(returns, columns=tickers)
        port = rp.Portfolio(returns=df)
        port.assets_stats(method_mu="hist", method_cov="ledoit")     # proven shrinkage covariance
        port.alpha = alpha                                           # CVaR confidence is a port attribute
        w = port.optimization(model="Classic", rm=rm, obj=obj, rf=0.0, l=0.0, hist=True)
        if w is None or w.empty:
            return None
        return {t: round(float(w.loc[t, "weights"]), 6) for t in tickers if t in w.index}
    except Exception:
        return None
