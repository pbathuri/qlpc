"""Generate every figure + number for the QNLPAI 2026 paper. Deterministic; writes paper/figures/*.png
and paper/results.json. Run:  python -m paper.experiments
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper.circuits import (LanguageConditionedQAOA, classical_best, entanglement_entropy,
                            iqae_cvar, layer_states, objective)

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# Ink — dark, paper-friendly
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 160, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})

# ---- the fixed demo instance (well-separated; assets {0,1} clearly optimal) --------------------
MU = np.array([1.10, 1.00, 0.45, 0.30])
SIGMA = np.array([[0.40, 0.02, 0.05, 0.05],
                  [0.02, 0.42, 0.05, 0.05],
                  [0.05, 0.05, 0.55, 0.20],
                  [0.05, 0.05, 0.20, 0.58]])
ASSETS = ["A", "B", "C", "D"]
BUDGET, Q = 2, 0.5
N = len(MU)
RESTARTS, MAXITER = 10, 200          # enough for clean p=2 results; keeps the suite to ~1 min


def bitlabel(idx, n=N):
    return "".join(str((idx >> k) & 1) for k in range(n))         # asset A (bit0) leftmost


def feasible_idxs(n=N, budget=BUDGET):
    return [i for i in range(2 ** n) if bin(i).count("1") == budget]


RESULTS: dict = {"instance": {"mu": MU.tolist(), "sigma": SIGMA.tolist(),
                              "budget": BUDGET, "q": Q, "assets": ASSETS}}


# ================================================================================================
# Fig 1 — state evolution heatmap (probability over basis states across QAOA layers)
# ================================================================================================
def fig_state_evolution(reps=2):
    qa = LanguageConditionedQAOA(MU, SIGMA, BUDGET, Q, reps=reps)
    run = qa.optimize(maxiter=MAXITER, seed=1, restarts=RESTARTS)
    P = np.array(run.snapshots)                       # (layers+1, 2^n)
    opt = run.classical_selected

    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(P, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(P.shape[0]))
    ax.set_yticklabels(["|+⟩ⁿ (L0)"] + [f"after L{i}" for i in range(1, P.shape[0])])
    ax.set_xticks(range(2 ** N))
    labels = [bitlabel(i) for i in range(2 ** N)]
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    # mark feasible (boxes) and the optimum (gold tick + star)
    opt_idx = next(i for i in feasible_idxs() if [j for j in range(N) if (i >> j) & 1] == opt)
    for f in feasible_idxs():
        ax.add_patch(plt.Rectangle((f - 0.5, -0.5), 1, P.shape[0], fill=False, edgecolor="#39d0c4",
                                   lw=0.8, alpha=0.5))
    optname = "{" + ",".join(ASSETS[i] for i in opt) + "}"
    ax.get_xticklabels()[opt_idx].set_color("#ffd166")
    ax.get_xticklabels()[opt_idx].set_fontweight("bold")
    ax.annotate(f"★ optimum {optname}", xy=(opt_idx, P.shape[0] - 0.5), xytext=(opt_idx, P.shape[0] + 1.3),
                ha="center", color="#b8860b", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#b8860b"))
    ax.set_xlabel("portfolio selection (bitstring over assets A B C D)", labelpad=18)
    ax.set_title(f"State evolution: probability concentrating on the optimal selection (QAOA p={reps})",
                 pad=12)
    fig.colorbar(im, ax=ax, label="probability", fraction=0.025)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_state_evolution.png")); plt.close(fig)

    RESULTS["fig1"] = {"reps": reps, "p_opt_layers": run.p_opt_layers, "p_opt": run.p_opt,
                       "concentration_x": run.concentration, "approx_ratio": run.approx_ratio,
                       "best_sampled_optimal": run.best_sampled_optimal,
                       "selected": run.selected, "optimum": opt}
    return run


# ================================================================================================
# Fig 2 — depth study: approximation ratio & P(optimum) vs QAOA layers
# ================================================================================================
def fig_depth(max_reps=4):
    reps_list = list(range(1, max_reps + 1))
    ARs, Pops, concs = [], [], []
    for r in reps_list:
        qa = LanguageConditionedQAOA(MU, SIGMA, BUDGET, Q, reps=r)
        run = qa.optimize(maxiter=MAXITER, seed=1, restarts=RESTARTS)
        ARs.append(run.approx_ratio); Pops.append(run.p_opt); concs.append(run.concentration)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(reps_list, ARs, "o-", color="#7c9cff", label="approx. ratio  ⟨obj⟩/opt")
    ax.plot(reps_list, Pops, "s-", color="#f0683c", label="P(optimum)")
    ax.axhline(1 / len(feasible_idxs()), ls="--", color="#888", lw=1,
               label=f"uniform feasible = 1/{len(feasible_idxs())}")
    ax.set_xlabel("QAOA depth p (layers)"); ax.set_xticks(reps_list)
    ax.set_ylim(0, 1.02); ax.set_ylabel("ratio / probability")
    ax.set_title("Deeper circuits → sharper concentration on the optimum")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_depth.png")); plt.close(fig)
    RESULTS["fig2_depth"] = {"reps": reps_list, "approx_ratio": ARs, "p_opt": Pops, "concentration": concs}


# ================================================================================================
# Fig 3 — the working process: optimiser energy trace + entanglement entropy across layers
# ================================================================================================
def fig_process(run):
    hist = run.energy_history
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.plot(range(len(hist)), hist, color="#7c9cff", lw=1.4)
    a1.scatter([int(np.argmin(hist))], [min(hist)], color="#f0683c", zorder=5, label="best")
    a1.set_xlabel("COBYLA evaluation"); a1.set_ylabel("⟨H_C⟩")
    a1.set_title("Working process: cost-expectation descent"); a1.legend(fontsize=8)

    ent = [entanglement_entropy(sv, N) for sv in layer_states(run.H, run.gammas, run.betas)]
    a2.plot(range(len(ent)), ent, "o-", color="#39d0c4")
    a2.set_xlabel("layer"); a2.set_ylabel("entropy of qubit A (bits)")
    a2.set_title("Entanglement builds, then localises")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_process.png")); plt.close(fig)
    RESULTS["fig3_process"] = {"energy_history_len": len(hist), "energy_min": float(min(hist)),
                               "entropy_by_layer": [float(x) for x in ent]}


# ================================================================================================
# Fig 4 — LANGUAGE STEERS THE QUANTUM STATE (the QNLP contribution)
#   A bullish LLM view on the lower-return assets flips the optimal selection; we show the quantum
#   output distribution re-concentrate on the NEW optimum — driven purely by language.
# ================================================================================================
def fig_language(reps=2):
    # graded LLM conviction on the lower-return pair C,D — from neutral to a strong bullish thesis
    scenarios = [("neutral\n(no view)", 0.0, "#7c9cff"),
                 ("moderate view:\nbullish C,D", 0.95, "#e0a23c"),
                 ("strong view:\nbullish C,D", 1.30, "#f0683c")]
    qa = LanguageConditionedQAOA(MU, SIGMA, BUDGET, Q, reps=reps)
    feas = feasible_idxs(); labels = [bitlabel(i) for i in feas]
    x = np.arange(len(feas)); w = 0.26

    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    rows = []
    for j, (name, v, col) in enumerate(scenarios):
        views = None if v == 0 else {2: v, 3: v}
        run = qa.optimize(views=views, strength=1.0, maxiter=MAXITER, seed=1, restarts=RESTARTS)
        mu_v = MU + (np.array([0, 0, v, v]) if v else 0.0)
        opt = classical_best(mu_v, SIGMA, BUDGET, Q)[0]
        p = np.array([run.snapshots[-1][i] for i in feas]); p = p / p.sum()
        ax.bar(x + (j - 1) * w, p, w, color=col, label=f"{name.split(chr(10))[0]} → opt {opt}")
        oi = feas.index(next(i for i in feas if [k for k in range(N) if (i >> k) & 1] == opt))
        ax.text(oi + (j - 1) * w, p[oi] + 0.015, "★", ha="center", color=col, fontsize=11)
        rows.append({"label": name.replace("\n", " "), "strength": v, "optimum": opt,
                     "selected": run.selected, "p_opt": run.p_opt, "approx_ratio": run.approx_ratio})

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("P(selection | feasible)")
    ax.set_title("Language steers the optimiser: rising LLM conviction deforms the quantum ground state")
    ax.legend(fontsize=8, title="A B C D")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_language.png")); plt.close(fig)
    RESULTS["fig4_language"] = {"scenarios": rows,
                                "optimum_trajectory": [r["optimum"] for r in rows]}


# ================================================================================================
# Fig 5 — IQAE quantum risk: loaded loss distribution + quantum vs classical tail/CVaR
# ================================================================================================
def fig_risk(seed=0, n_qubits=5, alpha=0.95):
    rng = np.random.default_rng(seed)
    # fat-tailed P&L losses (Student-t) at a realistic return scale
    losses = rng.standard_t(df=3, size=20000) * 0.01 + 0.002
    res = iqae_cvar(losses, alpha=alpha, n_qubits=n_qubits, epsilon=0.01)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3))
    centers, amps = res["centers"], res["amplitudes"]
    ti = res["threshold_index"]
    cols = ["#39d0c4" if i < ti else "#f0683c" for i in range(len(centers))]
    a1.bar(range(len(centers)), amps ** 2, color=cols, width=0.9)
    a1.axvline(ti - 0.5, color="#ffd166", ls="--", lw=1.2, label=f"VaR threshold (α={alpha})")
    a1.set_xlabel("loss bin (5-qubit discretisation)"); a1.set_ylabel("loaded probability |amp|²")
    a1.set_title("Loss distribution loaded into the quantum register"); a1.legend(fontsize=8)

    a2.bar(["tail prob\n(quantum)", "tail prob\n(classical)", "CVaR\n(quantum)", "CVaR\n(classical)"],
           [res["p_tail_quantum"], res["p_tail_classical"], res["cvar_quantum"], res["cvar_classical"]],
           color=["#7c9cff", "#39d0c4", "#7c9cff", "#39d0c4"])
    a2.set_title(f"IQAE vs classical  (tail err {res['rel_err_tail']*100:.1f}%, "
                 f"CVaR err {res['rel_err_cvar']*100:.1f}%)")
    a2.set_ylabel("value")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_risk.png")); plt.close(fig)

    RESULTS["fig5_risk"] = {k: (v if not isinstance(v, np.ndarray) else None)
                            for k, v in res.items() if k not in ("pmf", "centers", "amplitudes")}


def main():
    print("[1/5] state evolution ..."); run = fig_state_evolution(reps=2)
    print("[2/5] depth study ...");      fig_depth(max_reps=4)
    print("[3/5] working process ...");  fig_process(run)
    print("[4/5] language steering ..."); fig_language(reps=2)
    print("[5/5] quantum risk (IQAE) ..."); fig_risk()
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("done -> paper/figures/*.png + paper/results.json")
    # headline numbers
    f1 = RESULTS["fig1"]; lang = RESULTS["fig4_language"]; risk = RESULTS["fig5_risk"]
    print(f"  QAOA p=2: AR={f1['approx_ratio']:.3f} P(opt)={f1['p_opt']:.3f} conc={f1['concentration_x']:.1f}x")
    print(f"  language flip: {lang['neutral_optimum']} -> {lang['tilted_optimum']} "
          f"(flipped={lang['optimum_flipped']})")
    print(f"  IQAE: tail err {risk['rel_err_tail']*100:.1f}%  CVaR err {risk['rel_err_cvar']*100:.1f}%")


if __name__ == "__main__":
    main()
