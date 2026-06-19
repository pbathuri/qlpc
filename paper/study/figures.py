"""Honest, distribution-first figures for the study (no single-instance anecdotes).
Reads paper/study/data/study_results.json (local validation or aggregated HPC run).
Run:  python -m paper.study.figures
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(HERE, "data", "study_results.json")
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.25})


def save(fig, stem):
    fig.savefig(os.path.join(FIG, stem + ".png")); fig.savefig(os.path.join(FIG, stem + ".pdf")); plt.close(fig)


def main():
    res = json.load(open(RES)); R = res["rows"]; S = res["summary"]; C = res["config"]
    rank = np.array([r["Q1_rank_pct"] for r in R]) * 100
    conc = np.array([r["Q1_conc"] for r in R])
    changed = np.array([r["Q2_selection_changed"] for r in R])
    diff = np.array([r["Q3_ret_diff"] for r in R])[changed] * 100  # percent, where language acted

    # Q1 — optimizer quality across instances
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3))
    a1.hist(rank, bins=20, color="#7c9cff", edgecolor="#11151f")
    a1.axvline(np.median(rank), color="#f0683c", lw=2, label=f"median {np.median(rank):.0f}th pct")
    a1.set_title("Q1  QAOA pick: rank-percentile among feasible"); a1.set_xlabel("percentile (100=optimum)")
    a1.legend(fontsize=8)
    a2.hist(conc, bins=20, color="#39d0c4", edgecolor="#11151f")
    a2.axvline(conc.mean(), color="#f0683c", lw=2, label=f"mean {conc.mean():.1f}×")
    a2.set_title("Q1  concentration on optimum (× uniform)"); a2.set_xlabel("× uniform"); a2.legend(fontsize=8)
    fig.suptitle(f"Optimizer quality over {len(R)} instances (n={C['n']}, p={C['reps']}, {C['device']})")
    fig.tight_layout(); save(fig, "study_q1_quality")

    # Q3 — the honest headline: out-of-sample value of language tilt (only where it changed the portfolio)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    if len(diff) > 0:
        ax.hist(diff, bins=20, color="#9aa7bd", edgecolor="#11151f")
        m = diff.mean(); lo, hi = np.array(S["Q3_oos_diff_ci95_when_changed"]) * 100
        ax.axvline(m, color="#f0683c", lw=2, label=f"mean {m:+.2f}%  (95% CI [{lo:+.2f},{hi:+.2f}])")
        sig = "significant" if (lo > 0 or hi < 0) else "CI spans 0"
        ax.set_title(f"Q3 (PRELIMINARY — needs point-in-time news): OOS view−neutral, "
                     f"n={S['Q3_n_changed']}\n{S['Q3_pct_view_beats_neutral_when_changed']*100:.0f}% favor "
                     f"the view · {sig} · recent-news/historical-window mismatch not yet controlled")
        ax.legend(fontsize=8)
    ax.axvline(0, color="#888", lw=1); ax.set_xlabel("realized test-window return difference (%)")
    fig.tight_layout(); save(fig, "study_q3_oos_value")

    print("wrote study_q1_quality + study_q3_oos_value (png/pdf)")
    print("headline:", {k: S[k] for k in ("Q1_rank_pct_median", "Q1_pct_best_is_optimum",
                                          "Q2_pct_selection_changed", "Q3_oos_diff_mean_when_changed",
                                          "Q3_oos_diff_ci95_when_changed",
                                          "Q3_pct_view_beats_neutral_when_changed")})


if __name__ == "__main__":
    main()
