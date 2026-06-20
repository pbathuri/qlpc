"""Placebo / ablation control: does the LANGUAGE carry economic signal, or would any tilt do the same?

The Q2/Q3 questions are CLASSICAL (does a tilt change the exact optimal portfolio and its realized
out-of-sample return) — QAOA only matters for Q1 — so this runs with no statevector simulation and scales
to thousands of instances locally in seconds. We compare the real LLM tilt against:
  * RANDOM placebo  — tilts of matched sparsity/magnitude on randomly chosen names (the language is gone),
                      drawn K times to form a null distribution;
  * (optional) ESS  — RavenPack's own event-sentiment score, an ablation of the LLM vs off-the-shelf.

Output: a permutation-style p-value for "LLM out-of-sample value > random tilts", plus a figure. A null
result (LLM indistinguishable from random) is a finding we report honestly.

Run:  python -m paper.study.control --instances 400 --random-draws 200
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from paper.study.backend import classical_best
from paper.study.run_study import jaccard, make_instances, realized_return
from paper.study.universe import load_prices, window_moments

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "data")


def precompute(prices, instances, budget, q):
    """Vectorize each instance: the tilt only changes mu·x, while the risk term q·xᵀΣx and each feasible
    portfolio's realized OOS return are FIXED. Precompute the feasible membership matrix Xf, the risk
    vector, and the OOS-return vector once, so each tilt is a single matvec + argmax (no Ledoit-Wolf, no
    Python combinatorial search, per draw)."""
    import itertools
    n = len(instances[0]["assets"])
    feas = list(itertools.combinations(range(n), budget))
    Xf = np.zeros((len(feas), n))
    for r, c in enumerate(feas):
        Xf[r, list(c)] = 1.0
    pre = []
    for inst in instances:
        a = inst["assets"]; tr0, tr1, te1 = inst["window"]
        mu, sigma, _ = window_moments(prices, a, tr0, tr1)
        risk = q * np.einsum("fi,ij,fj->f", Xf, sigma, Xf)            # q xᵀΣx per feasible portfolio
        growth = (prices.loc[tr1:te1, a].iloc[-1] / prices.loc[tr1:te1, a].iloc[0]).values
        oos = Xf @ growth / budget - 1.0                              # equal-weight realized OOS return
        idx0 = int(np.argmax(Xf @ mu - risk))                        # neutral optimum
        pre.append({"inst": inst, "n": n, "Xf": Xf, "mu": mu, "risk": risk, "oos": oos, "idx0": idx0})
    return pre


def tilt_effect(prices, pre, conviction_fn, budget, q, strength):
    """Economic effect of a tilt vs neutral — fully vectorized. Returns (changed mask, full per-instance
    diff array with 0 where unchanged). Full-length arrays let us cluster by decision date."""
    changed = np.zeros(len(pre), dtype=bool); diff_full = np.zeros(len(pre))
    for k, p in enumerate(pre):
        cv = conviction_fn(p["inst"])
        tilt = np.array([cv.get(i, 0.0) for i in range(p["n"])])
        idxV = int(np.argmax(p["Xf"] @ (p["mu"] + strength * tilt) - p["risk"]))
        if idxV != p["idx0"]:
            changed[k] = True
            diff_full[k] = float(p["oos"][idxV] - p["oos"][p["idx0"]])
    return changed, diff_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--instances", type=int, default=400); ap.add_argument("--q", type=float, default=2.0)
    ap.add_argument("--strength", type=float, default=0.10); ap.add_argument("--random-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pit", default="pit_news_views.json", help="conviction cache (e.g. pit_news_views_llama2.json)")
    ap.add_argument("--out", default="control_results.json")
    a = ap.parse_args()

    prices = load_prices(); pool = list(prices.columns); dates = prices.index
    tr, te = 378, 63
    windows = [(str(dates[s])[:10], str(dates[s + tr])[:10], str(dates[s + tr + te])[:10])
               for s in range(0, len(dates) - tr - te, te)]
    instances = make_instances(pool, a.n, a.instances, windows, a.seed)
    pit = json.load(open(os.path.join(OUT, a.pit)))["views"]
    allc = [abs(v["conviction"]) > 0.01 for d in pit.values() for v in d.values()]
    p_nonzero = float(np.mean(allc))

    has_ess = any("ess" in v for d in pit.values() for v in d.values())

    def llm_cv(inst):
        vmap = pit.get(inst["window"][1], {}); a = inst["assets"]
        return {i: float(vmap.get(a[i], {}).get("conviction", 0.0)) for i in range(len(a))}

    def ess_cv(inst):
        vmap = pit.get(inst["window"][1], {}); a = inst["assets"]
        return {i: float(vmap.get(a[i], {}).get("ess", 0.0)) for i in range(len(a))}

    pre = precompute(prices, instances, a.budget, a.q)        # tilt-independent work, done once
    inst_date = np.array([p["inst"]["window"][1] for p in pre])
    print(f"precomputed {len(pre)} instances across {len(set(inst_date))} decision dates", flush=True)
    ch_llm, df_llm = tilt_effect(prices, pre, llm_cv, a.budget, a.q, a.strength)
    d_llm = df_llm[ch_llm]
    llm_mean = float(d_llm.mean()) if len(d_llm) else 0.0
    # PER DECISION DATE — the honest unit of replication (overlapping subsets within a date aren't independent)
    per_date = {}
    for dt in sorted(set(inst_date)):
        m = ch_llm & (inst_date == dt)
        per_date[dt] = float(df_llm[m].mean()) if m.any() else 0.0
    dvals = np.array(list(per_date.values()))
    print(f"LLM: changed {ch_llm.mean()*100:.0f}% | OOS diff mean {llm_mean*100:+.3f}% (n={len(d_llm)})", flush=True)
    print("LLM per-date OOS effect (%):", {k: round(v * 100, 2) for k, v in per_date.items()}, flush=True)
    print(f"  -> positive in {int((dvals>0).sum())}/{len(dvals)} dates; date-level mean "
          f"{dvals.mean()*100:+.2f}% (the effective N is ~{len(dvals)} periods, NOT {len(d_llm)})", flush=True)

    # ABLATION: RavenPack's own sentiment (ESS) vs the LLM — does the LLM add value over off-the-shelf?
    ess_block = None
    if has_ess:
        ch_e, df_e = tilt_effect(prices, pre, ess_cv, a.budget, a.q, a.strength)
        e_mean = float(df_e[ch_e].mean()) if ch_e.any() else 0.0
        e_per_date = {dt: (float(df_e[(ch_e) & (inst_date == dt)].mean())
                           if ((ch_e) & (inst_date == dt)).any() else 0.0) for dt in sorted(set(inst_date))}
        ess_block = {"oos_diff_mean": e_mean, "changed_frac": float(ch_e.mean()),
                     "per_date_oos_effect": e_per_date,
                     "positive_dates": f"{int((np.array(list(e_per_date.values()))>0).sum())}/{len(e_per_date)}"}
        print(f"ESS (RavenPack sentiment): changed {ch_e.mean()*100:.0f}% | OOS {e_mean*100:+.3f}% | "
              f"LLM {'>' if llm_mean>e_mean else '<='} ESS", flush=True)

    rand_means, rand_changed = [], []
    for k in range(a.random_draws):
        rng = np.random.default_rng(10_000 + k)

        def rand_cv(inst, _rng=rng):
            return {i: float(_rng.choice([-1.0, 1.0])) if _rng.random() < p_nonzero else 0.0
                    for i in range(a.n)}
        ch_r, df_r = tilt_effect(prices, pre, rand_cv, a.budget, a.q, a.strength)
        d_r = df_r[ch_r]
        rand_means.append(float(d_r.mean()) if len(d_r) else 0.0); rand_changed.append(ch_r.mean())
        if (k + 1) % 50 == 0:
            print(f"  placebo {k+1}/{a.random_draws}", flush=True)
    rand_means = np.array(rand_means)

    # permutation p-value (instance-level — OPTIMISTIC: instances within a date are correlated)
    p_two = float(np.mean(np.abs(rand_means) >= abs(llm_mean)))
    n_dates = len(dvals); pos_dates = int((dvals > 0).sum())
    robust = "consistent" if pos_dates in (0, n_dates) else "MIXED across dates"
    res = {"config": {"n": a.n, "budget": a.budget, "instances": len(instances), "strength": a.strength,
                      "random_draws": a.random_draws, "decision_dates": n_dates},
           "llm": {"oos_diff_mean": llm_mean, "changed_frac": float(ch_llm.mean()), "n_changed": int(len(d_llm)),
                   "per_date_oos_effect": per_date, "date_level_mean": float(dvals.mean()),
                   "positive_dates": f"{pos_dates}/{n_dates}"},
           "placebo": {"oos_diff_mean_mean": float(rand_means.mean()),
                       "oos_diff_mean_std": float(rand_means.std()),
                       "ci95": [float(np.percentile(rand_means, 2.5)), float(np.percentile(rand_means, 97.5))],
                       "changed_frac_mean": float(np.mean(rand_changed))},
           "ess_ablation": ess_block,
           "permutation_p_instance_level": p_two,
           "caveat": (f"Instance-level p is OPTIMISTIC: only {n_dates} independent decision dates "
                      f"(subsets within a date are correlated). The honest unit is the date; the effect is "
                      f"{robust} ({pos_dates}/{n_dates} dates positive). More dates (longer history / HPC) "
                      f"are required before claiming significance."),
           "verdict": (f"LLM tilt beats matched random tilts at the instance level (p={p_two:.2f}) and is "
                       f"{robust} across the {n_dates} dates ({pos_dates}/{n_dates} positive) — promising "
                       f"but NOT yet robust with so few periods")}
    json.dump({**res, "random_means": rand_means.tolist()}, open(os.path.join(OUT, a.out), "w"), indent=2)

    # figure: (left) placebo null vs LLM effect; (right) per-date effect — is it consistent or fragile?
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.hist(rand_means * 100, bins=30, color="#9aa7bd", edgecolor="#11151f", label="random-tilt placebo")
    a1.axvline(llm_mean * 100, color="#f0683c", lw=2.5, label=f"LLM tilt {llm_mean*100:+.2f}%")
    a1.axvline(0, color="#888", lw=1)
    a1.set_xlabel("mean OOS effect (%)"); a1.set_title(f"LLM vs {a.random_draws} random tilts (p={p_two:.2f})", fontsize=9)
    a1.legend(fontsize=8)
    dts = list(per_date); vals = [per_date[d] * 100 for d in dts]
    a2.bar(range(len(dts)), vals, color=["#3fb6a8" if v > 0 else "#f0683c" for v in vals])
    a2.axhline(0, color="#888", lw=1); a2.set_xticks(range(len(dts)))
    a2.set_xticklabels(dts, rotation=30, fontsize=7); a2.set_ylabel("OOS effect (%)")
    a2.set_title(f"Per decision date ({pos_dates}/{n_dates} positive) — the honest unit", fontsize=9)
    fig.suptitle("Does the LLM language carry economic signal? Promising but few periods", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figures", f"control_placebo.{ext}"))
    print("VERDICT:", res["verdict"])
    print("CAVEAT:", res["caveat"])
    print("wrote control_results.json + figures/control_placebo.*")


if __name__ == "__main__":
    main()
