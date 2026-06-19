"""Assemble a single-page draft poster PDF from the generated figures + headline results.
Run after experiments.py:  python -m paper.poster  ->  paper/poster.pdf
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
R = json.load(open(os.path.join(HERE, "results.json")))

BG, INK, ACC, HI = "#0b0e14", "#e6e9ef", "#7c9cff", "#f0683c"


def panel(ax, name):
    ax.imshow(mpimg.imread(os.path.join(FIG, name))); ax.axis("off")


def main():
    fig = plt.figure(figsize=(16.5, 23.4), facecolor=BG)        # A2 portrait
    gs = GridSpec(6, 2, figure=fig, height_ratios=[1.15, 0.5, 1.25, 1.25, 1.25, 0.55],
                  hspace=0.18, wspace=0.06, left=0.04, right=0.96, top=0.97, bottom=0.03)

    # --- title band ---
    t = fig.add_subplot(gs[0, :]); t.axis("off"); t.set_facecolor(BG)
    t.text(0.5, 0.78, "Watching Language Steer a Quantum Optimizer", ha="center", va="center",
           color=INK, fontsize=30, fontweight="bold")
    t.text(0.5, 0.5, "State-Evolution Visualization of LLM-Conditioned QAOA for\n"
                     "Cardinality-Constrained Portfolio Selection", ha="center", va="center",
           color=ACC, fontsize=18)
    t.text(0.5, 0.16, "QNLPAI 2026  ·  Quantum Computing for AI — Quantum Optimization (QNLP bridge)  ·  "
                      "QLPC Project", ha="center", va="center", color="#8a93a3", fontsize=13)

    # --- problem / approach text ---
    txt = fig.add_subplot(gs[1, :]); txt.axis("off")
    f1, lang, risk = R["fig1"], R["fig4_language"], R["fig5_risk"]
    traj = " → ".join("{" + ",".join("ABCD"[i] for i in o) + "}" for o in lang["optimum_trajectory"])
    txt.text(0.01, 0.5,
             "PROBLEM   QAOA is reported as a black box — one bitstring, one ratio — hiding HOW the state\n"
             "decides, with no port for the language-shaped views that drive real allocation.\n\n"
             "APPROACH   Instrument the quantum core of QLPC on IBM Qiskit so every state is observable:\n"
             "(1) snapshot the statevector each QAOA layer;  (2) encode an LLM natural-language view as a\n"
             "tilt on expected returns that deforms the cost Hamiltonian;  (3) IQAE for tail / CVaR risk.",
             ha="left", va="center", color=INK, fontsize=13, family="monospace")

    panel(fig.add_subplot(gs[2, 0]), "fig1_state_evolution.png")
    panel(fig.add_subplot(gs[2, 1]), "fig4_language.png")
    panel(fig.add_subplot(gs[3, 0]), "fig2_depth.png")
    panel(fig.add_subplot(gs[3, 1]), "fig3_process.png")
    panel(fig.add_subplot(gs[4, 0]), "fig5_risk.png")

    # --- results box (right of fig5) ---
    rb = fig.add_subplot(gs[4, 1]); rb.axis("off")
    lines = [
        ("RESULTS  (exact statevector, fixed seeds)", HI, 15, "bold"),
        (f"• State concentrates on the optimum: P(opt) 0.062 → {f1['p_opt']:.2f} at p=2", INK, 13, "normal"),
        (f"   = {f1['concentration_x']:.1f}× a uniform guess;  approx-ratio {f1['approx_ratio']:.2f};", INK, 13, "normal"),
        ("   exact optimum recovered by best sample.", INK, 13, "normal"),
        (f"• Language steers the ground state: optimum {traj}", ACC, 13, "bold"),
        ("   QAOA modal selection tracks it in all 3 regimes.", INK, 13, "normal"),
        (f"• IQAE tail-prob within {risk['rel_err_tail']*100:.2f}% of the loaded", INK, 13, "normal"),
        ("   distribution (the quantum-estimated quantity).", INK, 13, "normal"),
        ("", INK, 8, "normal"),
        ("HONEST LIMITS  small, well-separated, simulated", "#8a93a3", 11, "italic"),
        ("instances; no scaling/supremacy claim; additive tilt.", "#8a93a3", 11, "italic"),
    ]
    y = 0.97
    for s, c, fs, w in lines:
        style = "italic" if w == "italic" else "normal"
        wt = "bold" if w == "bold" else "normal"
        rb.text(0.0, y, s, ha="left", va="top", color=c, fontsize=fs, family="monospace",
                fontstyle=style, fontweight=wt)
        y -= 0.092

    # --- footer ---
    ft = fig.add_subplot(gs[5, :]); ft.axis("off")
    ft.text(0.5, 0.5, "Reproducible:  python -m paper.experiments   (Qiskit 2.x + Aer)   ·   "
                      "circuits.py · experiments.py · results.json",
            ha="center", va="center", color="#8a93a3", fontsize=12, family="monospace")

    fig.savefig(os.path.join(HERE, "poster.pdf"), facecolor=BG, bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "poster.png"), facecolor=BG, bbox_inches="tight", dpi=110)
    print("wrote paper/poster.pdf + paper/poster.png")


if __name__ == "__main__":
    main()
