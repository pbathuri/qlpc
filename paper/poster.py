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
    fig = plt.figure(figsize=(24, 36), facecolor=BG)            # QNLPAI max poster size 24" x 36" portrait
    gs = GridSpec(6, 2, figure=fig, height_ratios=[1.05, 0.92, 1.30, 1.30, 0.78, 0.45],
                  hspace=0.22, wspace=0.06, left=0.04, right=0.96, top=0.97, bottom=0.03)

    # --- title band ---
    t = fig.add_subplot(gs[0, :]); t.axis("off"); t.set_facecolor(BG)
    t.text(0.5, 0.78, "Watching Language Steer a Quantum Optimizer", ha="center", va="center",
           color=INK, fontsize=44, fontweight="bold")
    t.text(0.5, 0.5, "State-Evolution Visualization of LLM-Conditioned QAOA for\n"
                     "Cardinality-Constrained Portfolio Selection", ha="center", va="center",
           color=ACC, fontsize=26)
    t.text(0.5, 0.16, "QNLPAI 2026  ·  Quantum Computing for AI — Quantum Optimization (QNLP bridge)  ·  "
                      "QLPC Project", ha="center", va="center", color="#8a93a3", fontsize=19)

    # --- problem / approach text (two objectives; risk module is out of scope here) ---
    txt = fig.add_subplot(gs[1, :]); txt.axis("off")
    f1, lang = R["fig1"], R["fig4_language"]
    traj = " → ".join("{" + ",".join("ABCD"[i] for i in o) + "}" for o in lang["optimum_trajectory"])
    txt.text(0.01, 0.5,
             "PROBLEM   QAOA is reported as a black box — one bitstring, one ratio — hiding HOW the state\n"
             "decides, with no port for the language-shaped views that drive real allocation.\n\n"
             "GAP   LLM views exist for CLASSICAL Black-Litterman (Lee+ 2025); quantum views exist but are\n"
             "QML-predicted numbers solved by VQE (Chen+ 2023) — no language, no view of the quantum state.\n\n"
             "TWO OBJECTIVES   O1  encode an LLM natural-language view as a tilt on expected returns that\n"
             "deforms the QAOA cost Hamiltonian;   O2  snapshot the statevector each layer to make the\n"
             "language-steering auditable.   (Exact statevector simulation; IBM Qiskit.)",
             ha="left", va="center", color=INK, fontsize=18, family="monospace")

    panel(fig.add_subplot(gs[2, 0]), "fig1_state_evolution.png")   # O2: state evolution
    panel(fig.add_subplot(gs[2, 1]), "fig4_language.png")          # O1: language steering
    panel(fig.add_subplot(gs[3, 0]), "fig2_depth.png")             # support: depth
    panel(fig.add_subplot(gs[3, 1]), "fig3_process.png")           # support: working process

    # --- results box (full width, replaces the dropped risk panel) ---
    rb = fig.add_subplot(gs[4, :]); rb.axis("off")
    lines = [
        ("RESULTS  (exact statevector, fixed seeds)", HI, 24, "bold"),
        (f"O2  State concentrates on the optimum:  P(opt) 0.062 → {f1['p_opt']:.2f} at p=2  "
         f"= {f1['concentration_x']:.1f}× a uniform guess;  approx-ratio {f1['approx_ratio']:.2f};  "
         "exact optimum recovered.", INK, 19, "normal"),
        (f"O1  Language steers the ground state:  optimum {traj}  as LLM conviction rises;  "
         "the QAOA modal selection tracks it in all 3 regimes.", ACC, 19, "bold"),
        ("", INK, 10, "normal"),
        ("HONEST LIMITS   small (n=4), well-separated, exactly-simulated instances;  no scaling / hardware /"
         " quantum-advantage claim (brute force is the oracle);  additive language tilt.", "#8a93a3", 16, "italic"),
    ]
    y = 0.95
    for s, c, fs, w in lines:
        style = "italic" if w == "italic" else "normal"
        wt = "bold" if w == "bold" else "normal"
        rb.text(0.0, y, s, ha="left", va="top", color=c, fontsize=fs, family="monospace",
                fontstyle=style, fontweight=wt, wrap=True)
        y -= 0.16

    # --- footer ---
    ft = fig.add_subplot(gs[5, :]); ft.axis("off")
    ft.text(0.5, 0.5, "Reproducible:  python -m paper.experiments   (Qiskit 2.x + Aer)   ·   "
                      "circuits.py · experiments.py · results.json",
            ha="center", va="center", color="#8a93a3", fontsize=17, family="monospace")

    fig.savefig(os.path.join(HERE, "poster.pdf"), facecolor=BG, bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "poster.png"), facecolor=BG, bbox_inches="tight", dpi=110)
    print("wrote paper/poster.pdf + paper/poster.png")


if __name__ == "__main__":
    main()
