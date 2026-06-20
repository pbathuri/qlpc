"""Draft poster (<= 24in x 36in portrait) for the cautionary study. Run after the figures exist:
    python -m paper.study.poster   ->  paper/study/poster.pdf + poster.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
BG, INK, ACC, HI, OK = "#0b0e14", "#e6e9ef", "#7c9cff", "#f0683c", "#3fb6a8"
S = json.load(open(os.path.join(HERE, "data", "study_results.json")))["summary"]
Q = json.load(open(os.path.join(HERE, "data", "control_results.json")))            # qwen2.5
L = json.load(open(os.path.join(HERE, "data", "control_llama2.json")))             # Llama 2
qp = (Q["llm"]["oos_diff_mean"] - Q["placebo"]["oos_diff_mean_mean"]) * 100
lp = (L["llm"]["oos_diff_mean"] - L["placebo"]["oos_diff_mean_mean"]) * 100


def panel(ax, name):
    ax.imshow(mpimg.imread(os.path.join(FIG, name))); ax.axis("off")


def main():
    fig = plt.figure(figsize=(24, 36), facecolor=BG)
    gs = GridSpec(6, 1, figure=fig, height_ratios=[1.0, 0.95, 1.5, 1.25, 1.05, 0.4],
                  hspace=0.16, left=0.04, right=0.96, top=0.97, bottom=0.03)

    t = fig.add_subplot(gs[0]); t.axis("off")
    t.text(0.5, 0.74, "Does Language Really Help a Quantum Portfolio Optimizer?", ha="center",
           color=INK, fontsize=46, fontweight="bold")
    t.text(0.5, 0.42, "A Bias-Controlled Study and a Training-Horizon Confound in LLM-Conditioned QAOA",
           ha="center", color=ACC, fontsize=27)
    t.text(0.5, 0.12, "QNLPAI 2026 Student Poster Session  ·  author details withheld for double-blind review",
           ha="center", color="#8a93a3", fontsize=19)

    txt = fig.add_subplot(gs[1]); txt.axis("off")
    txt.text(0.0, 0.5,
             "QUESTION   When a backtest says 'LLM news-views help a quantum portfolio optimizer,' is that\n"
             "signal — or an artifact?  We build a deliberately bias-controlled pipeline to catch ourselves.\n\n"
             "PIPELINE   Objective universe (Dow 30, CRSP total returns) -> an LLM reads RavenPack headlines\n"
             "time-stamped BEFORE each decision date into per-asset convictions -> a +-10% view tilts the QAOA\n"
             "cost Hamiltonian -> hold 5 of 14, evaluate the held-out next quarter.  CONTROLS: matched random-\n"
             "tilt placebo; ablation vs RavenPack's own sentiment; and TWO LLMs on the SAME headlines —\n"
             "qwen2.5-14B (training overlaps 2024) vs Llama 2 (Sept-2022 cutoff, genuinely out-of-sample).",
             ha="left", va="center", color=INK, fontsize=19, family="monospace")

    panel(fig.add_subplot(gs[2]), "compare_clean_vs_contaminated.png")    # headline
    panel(fig.add_subplot(gs[3]), "study_q1_quality.png")                 # QAOA quality

    rb = fig.add_subplot(gs[4]); rb.axis("off")
    lines = [
        ("KEY FINDING — the apparent 'language signal' is the LLM's hindsight", HI, 26, "bold"),
        (f"   qwen2.5 (train overlaps 2024):  +{qp:.2f}% OOS premium over random tilts (p={Q['permutation_p_instance_level']:.2f})",
         INK, 20, "normal"),
        (f"   Llama 2 (clean, pre-2023):       {lp:+.2f}% (p={L['permutation_p_instance_level']:.2f}) — indistinguishable from random",
         OK, 20, "bold"),
        ("   => the out-of-sample 'alpha' VANISHES with a genuinely out-of-sample LLM.", INK, 20, "normal"),
        ("", INK, 8, "normal"),
        (f"SUPPORTING   language has real leverage (changes {S['Q2_pct_selection_changed']*100:.0f}% of optima), but shallow QAOA is a",
         INK, 20, "normal"),
        (f"   WEAK exact optimizer — finds the true optimum only {S['Q1_pct_best_is_optimum']*100:.1f}% of the time. No quantum-advantage claim.",
         INK, 20, "normal"),
        ("", INK, 8, "normal"),
        ("HONEST LIMITS  only 4 decision dates (instance-level p pseudo-replicated); small-n exact sim, no",
         "#8a93a3", 17, "italic"),
        ("   hardware; large-cap US; no transaction costs; additive tilt. None inflate the central result.",
         "#8a93a3", 17, "italic"),
        ("CONTRIBUTION  a reproducible bias-control template + a quantified warning: date your LLM against",
         ACC, 19, "bold"),
        ("   your backtest, or training-horizon look-ahead will manufacture alpha that isn't there.", ACC, 19, "bold"),
    ]
    y = 0.98
    for s, c, fs, w in lines:
        rb.text(0.0, y, s, ha="left", va="top", color=c, fontsize=fs, family="monospace",
                fontweight="bold" if w == "bold" else "normal", fontstyle="italic" if w == "italic" else "normal")
        y -= 0.083

    ft = fig.add_subplot(gs[5]); ft.axis("off")
    ft.text(0.5, 0.5, "Reproducible: python -m paper.study.run_study / control / compare_figure  ·  fixed seeds, "
                      "committed CRSP + conviction caches  ·  scales on HPC (SLURM/GPU)",
            ha="center", color="#8a93a3", fontsize=17, family="monospace")

    fig.savefig(os.path.join(HERE, "poster.pdf"), facecolor=BG, bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "poster.png"), facecolor=BG, bbox_inches="tight", dpi=100)
    print(f"wrote poster (qwen +{qp:.2f}% -> Llama2 {lp:+.2f}%)")


if __name__ == "__main__":
    main()
