"""The cautionary headline figure: an apparent LLM 'signal' is training-horizon hindsight.

Compares the language-specific out-of-sample premium (LLM tilt minus matched random tilts) for a model
whose training overlaps the evaluation window (qwen2.5-14B, released 2024) vs a genuinely out-of-sample
model (Llama 2, Sept-2022 cutoff). Reads control_results.json (qwen2.5) + control_llama2.json (Llama 2).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")


def load(f):
    d = json.load(open(os.path.join(HERE, "data", f)))
    return d["llm"], d["placebo"], d["permutation_p_instance_level"]


def main():
    ql, qp, qpp = load("control_results.json")           # qwen2.5 (contaminated)
    ll, lp, lpp = load("control_llama2.json")            # Llama 2 (clean)
    q_prem = (ql["oos_diff_mean"] - qp["oos_diff_mean_mean"]) * 100
    l_prem = (ll["oos_diff_mean"] - lp["oos_diff_mean_mean"]) * 100

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.7))

    # Panel A — language-specific premium over random, with significance
    labels = ["qwen2.5-14B\n(train overlaps 2024)", "Llama 2\n(Sept-2022 cutoff)"]
    prem = [q_prem, l_prem]; ps = [qpp, lpp]
    bars = a1.bar(labels, prem, color=["#f0683c", "#3fb6a8"], edgecolor="#11151f")
    a1.axhline(0, color="#888", lw=1)
    for b, p, v in zip(bars, ps, prem):
        a1.text(b.get_x() + b.get_width() / 2, v + (0.04 if v >= 0 else -0.08),
                f"{v:+.2f}%\n(p={p:.2f})", ha="center", fontsize=9,
                color="#11151f", fontweight="bold")
    a1.set_ylabel("OOS premium over random tilts (%)")
    a1.set_title("Apparent 'language signal' is the LLM's hindsight", fontsize=10)
    a1.set_ylim(min(prem) - 0.4, max(prem) + 0.5)

    # Panel B — per-decision-date OOS effect, both models
    dts = list(ql["per_date_oos_effect"]); x = np.arange(len(dts)); w = 0.38
    qv = [ql["per_date_oos_effect"][d] * 100 for d in dts]
    lv = [ll["per_date_oos_effect"][d] * 100 for d in dts]
    a2.bar(x - w / 2, qv, w, color="#f0683c", label="qwen2.5 (contaminated)")
    a2.bar(x + w / 2, lv, w, color="#3fb6a8", label="Llama 2 (clean)")
    a2.axhline(0, color="#888", lw=1); a2.set_xticks(x)
    a2.set_xticklabels(dts, rotation=30, fontsize=7); a2.set_ylabel("OOS effect (%)")
    a2.set_title("Per decision date (only 4 — not robust either way)", fontsize=10)
    a2.legend(fontsize=8)

    fig.suptitle("Out-of-sample value of LLM-conditioned tilts collapses with a genuinely out-of-sample LLM",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"compare_clean_vs_contaminated.{ext}"))
    print(f"qwen2.5 premium {q_prem:+.2f}% (p={qpp:.2f}) -> Llama2 premium {l_prem:+.2f}% (p={lpp:.2f})")
    print("wrote figures/compare_clean_vs_contaminated.*")


if __name__ == "__main__":
    main()
