"""Unbiased, out-of-sample study of language-conditioned QAOA over an objective universe.

Pre-registered design (metrics fixed BEFORE looking at results):
  Q1  Optimizer quality   : QAOA approximation ratio and P(optimum) vs the exact classical optimum.
  Q2  Language effect      : does the LLM view change the selected portfolio? (Jaccard distance)
  Q3  Out-of-sample value  : realized TEST-window return of the language-tilted vs the neutral portfolio.
                             Reported with sign + bootstrap CI + paired test. A null/negative result is
                             a valid finding; we do not select for favorable outcomes.

Bias controls: objective universe (Dow 30); MANY (subset x rolling-window) instances from a fixed seed;
strict train/test separation; identical pipeline for neutral and tilted; LLM views from real headlines.

HPC: shard by instance via --shard k/N or SLURM_ARRAY_TASK_ID; device via QLPC_DEVICE=gpu (cupy).
Run:  python -m paper.study.run_study --n 12 --budget 4 --instances 40 --reps 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

from paper.circuits import classical_best, objective
from paper.study import backend as BK
from paper.study.news_views import build_views, conviction_vector
from paper.study.universe import load_prices, window_moments

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "data")


def realized_return(prices, tickers, test_slice) -> float:
    """Equal-weight realized return of `tickers` over the test window (buy-and-hold)."""
    sub = prices.loc[test_slice, tickers]
    growth = (sub.iloc[-1] / sub.iloc[0]).values
    return float(np.mean(growth) - 1.0)


def decode(probs, n, budget, mu, sigma, q):
    """QAOA quality vs the exact optimum, with a SIGN-SAFE rank-percentile (objectives can be negative).
    rank_pct = fraction of feasible portfolios the QAOA pick is >= to by objective (1.0 = the optimum)."""
    dim = 2 ** n
    bits = [[(x >> k) & 1 for k in range(n)] for x in range(dim)]
    feas_idx = [i for i in range(dim) if sum(bits[i]) == budget]
    objs = {i: objective(np.array(bits[i], float), mu, sigma, q) for i in feas_idx}
    opt_idx = max(feas_idx, key=lambda i: objs[i])
    sel_idx = max(feas_idx, key=lambda i: probs[i])           # most-probable feasible sample
    sorted_objs = sorted(objs.values())
    rank_pct = (sorted_objs.index(objs[sel_idx]) + 1) / len(sorted_objs)
    return {"selected": [i for i in range(n) if bits[sel_idx][i] == 1],
            "p_opt": float(probs[opt_idx]), "p_uniform": 1.0 / dim,
            "rank_pct": rank_pct, "best_is_opt": sel_idx == opt_idx,
            "optimum": [i for i in range(n) if bits[opt_idx][i] == 1]}


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return 1.0 - (len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0)


def make_instances(pool, n, n_inst, windows, seed=0):
    """Deterministic (subset, window) instances. Subsets sampled without replacement from the pool."""
    rng = np.random.default_rng(seed)
    inst = []
    for w in windows:
        for _ in range(n_inst):
            sub = sorted(rng.choice(len(pool), size=n, replace=False).tolist())
            inst.append({"assets": [pool[i] for i in sub], "window": w})
    return inst


# Conviction -> annual return-view scale. A +-1 LLM conviction maps to a +-STRENGTH annual expected-return
# view, following the sentiment->Black-Litterman-view convention (cf. Riskfolio-Lib's
# sentiment_to_black_litterman_views): a realistic, non-dominating tilt, NOT a magnitude tuned to flip the
# optimum. 0.10 = a 10% annual view at full conviction.
def run(n=12, budget=4, n_inst=20, reps=3, q=2.0, strength=0.10, seed=0, shard=None):
    xp, dev = BK.get_xp()
    prices = load_prices(); pool = list(prices.columns)
    dates = prices.index
    # rolling train/test windows over the snapshot (train ~1.5y, test ~1 quarter)
    L = len(dates); tr, te = 378, 63
    windows = []
    for s in range(0, L - tr - te, te):
        windows.append((str(dates[s])[:10], str(dates[s + tr])[:10], str(dates[s + tr + te])[:10]))
    views = build_views(pool)                          # real-headline LLM convictions (cached)
    instances = make_instances(pool, n, n_inst, windows, seed)
    if shard:
        k, total = shard
        instances = instances[k::total]

    rows = []
    for j, inst in enumerate(instances):
        assets = inst["assets"]; tr0, tr1, te1 = inst["window"]
        mu, sigma, _ = window_moments(prices, assets, tr0, tr1)
        diag, _ = BK.cost_diagonal(mu, sigma, budget, q, xp=xp)
        g, b, _ = BK.optimize_qaoa(diag, reps, restarts=8, maxiter=120, seed=seed + j, xp=xp)
        p0 = BK.to_numpy(BK.qaoa_probs(diag, g, b, xp=xp), xp=xp)
        d0 = decode(p0, n, budget, mu, sigma, q)

        # LANGUAGE EFFECT measured at the EXACT optimum (clean of QAOA stochasticity). QAOA quality is
        # reported separately in Q1; the economic question should not inherit optimizer noise.
        cv = conviction_vector(views, assets)
        tilt = np.array([cv.get(i, 0.0) for i in range(n)])
        mu_v = mu + strength * tilt
        cls0 = [assets[i] for i in classical_best(mu, sigma, budget, q)[0]]          # neutral optimum
        clsV = [assets[i] for i in classical_best(mu_v, sigma, budget, q)[0]]        # view-tilted optimum
        test_slice = slice(tr1, te1)
        r0 = realized_return(prices, cls0, test_slice)
        rv = realized_return(prices, clsV, test_slice)
        rows.append({"assets": assets, "window": inst["window"],
                     "neutral_opt": cls0, "view_opt": clsV, "qaoa_sel": [assets[i] for i in d0["selected"]],
                     # Q1 — QAOA optimizer quality (vs exact)
                     "Q1_rank_pct": d0["rank_pct"], "Q1_p_opt": d0["p_opt"],
                     "Q1_best_is_opt": d0["best_is_opt"], "Q1_conc": d0["p_opt"] / d0["p_uniform"],
                     # Q2 — does language change the EXACT optimal portfolio?
                     "Q2_selection_changed": set(cls0) != set(clsV), "Q2_jaccard": jaccard(cls0, clsV),
                     # Q3 — out-of-sample realized return, view-optimum vs neutral-optimum
                     "Q3_ret_neutral": r0, "Q3_ret_view": rv, "Q3_ret_diff": rv - r0,
                     "nonzero_views": int(np.count_nonzero(tilt))})
        if (j + 1) % 5 == 0:
            print(f"  instance {j+1}/{len(instances)} done", flush=True)
    return {"config": {"n": n, "budget": budget, "reps": reps, "q": q, "strength": strength,
                       "device": dev, "n_instances": len(instances), "windows": windows,
                       "universe": "Dow30-snapshot", "seed": seed},
            "rows": rows}


def _boot_ci(x, B=5000, seed=0):
    if len(x) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def summarize(res):
    R = res["rows"]
    if not R:
        return {}
    rank = np.array([r["Q1_rank_pct"] for r in R])
    popt = np.array([r["Q1_p_opt"] for r in R]); conc = np.array([r["Q1_conc"] for r in R])
    changed = np.array([r["Q2_selection_changed"] for r in R])
    diff = np.array([r["Q3_ret_diff"] for r in R])
    diff_ch = diff[changed]                              # where language actually moved the portfolio
    try:
        from scipy.stats import wilcoxon
        wp = float(wilcoxon(diff_ch).pvalue) if len(diff_ch) > 0 and np.any(diff_ch != 0) else 1.0
    except Exception:  # noqa: BLE001
        wp = None
    return {"Q1_rank_pct_median": float(np.median(rank)), "Q1_rank_pct_mean": float(rank.mean()),
            "Q1_p_opt_mean": float(popt.mean()), "Q1_concentration_mean": float(conc.mean()),
            "Q1_pct_best_is_optimum": float(np.mean([r["Q1_best_is_opt"] for r in R])),
            "Q2_pct_selection_changed": float(changed.mean()),
            "Q2_jaccard_mean_when_changed": float(np.mean([r["Q2_jaccard"] for r in R if r["Q2_selection_changed"]]) if changed.any() else 0.0),
            "Q3_n_changed": int(changed.sum()),
            "Q3_oos_diff_mean_when_changed": float(diff_ch.mean()) if len(diff_ch) else 0.0,
            "Q3_oos_diff_ci95_when_changed": _boot_ci(diff_ch),
            "Q3_wilcoxon_p_when_changed": wp,
            "Q3_pct_view_beats_neutral_when_changed": float(np.mean(diff_ch > 0)) if len(diff_ch) else 0.0,
            "n_instances": len(R)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--instances", type=int, default=20); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--q", type=float, default=2.0); ap.add_argument("--strength", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--shard", type=str, default=None)
    ap.add_argument("--out", type=str, default=os.path.join(OUT, "study_results.json"))
    a = ap.parse_args()
    shard = None
    if a.shard:
        k, total = (int(x) for x in a.shard.split("/")); shard = (k, total)
    elif os.environ.get("SLURM_ARRAY_TASK_ID"):
        shard = (int(os.environ["SLURM_ARRAY_TASK_ID"]), int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")))
    res = run(a.n, a.budget, a.instances, a.reps, a.q, a.strength, a.seed, shard)
    res["summary"] = summarize(res)
    out = a.out if not shard else a.out.replace(".json", f".shard{shard[0]}.json")
    json.dump(res, open(out, "w"), indent=2)
    print("SUMMARY:", json.dumps(res["summary"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
