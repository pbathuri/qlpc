# Research-grade study — design, bias controls, and honest findings

This study replaces the earlier hand-picked, single-instance demo with an unbiased, out-of-sample
evaluation over an objective universe, designed to survive a hostile peer review. It is HPC-ready
(BigRed200 + A100 via cupy); locally we validate the pipeline at moderate scale on CPU.

## Pipeline (architected workflow)
`universe.py` (objective Dow-30 snapshot, WRDS/CRSP in production) → `news_views.py` (a local LLM reads
**real headlines** → per-asset conviction) → `backend.py` (device-agnostic fast QAOA statevector;
CPU/numpy or GPU/cupy) → `run_study.py` (many subset×window instances, strict train/test, pre-declared
metrics + statistics) → `figures.py`.

## Bias controls (what each removes)
- **Universe-selection bias** → objective index membership (Dow 30), not author-chosen names.
- **Instance cherry-picking** → hundreds of `(subset × rolling-window)` instances from a fixed seed; we
  report **distributions**, never a favorable single case.
- **Parameter cherry-picking** → `(budget, q)` fixed a priori; the view magnitude is set to a realistic
  ±10% annual return view (the sentiment→Black–Litterman convention, cf. Riskfolio-Lib), **not** tuned to
  make the optimum flip.
- **Language-input bias** → views come from **real published headlines via a local LLM**, not text we
  wrote. ~Most names read neutral (honest: real news is noisy).
- **Result-selection bias** → metrics pre-declared; null/negative results reported as findings.
- **Look-ahead bias** → strict train→test separation; QAOA quality (Q1) and language effect (Q2) measured
  on the train window, value (Q3) on the held-out test window.

## Metrics (pre-registered)
- **Q1 Optimizer quality** — QAOA pick's rank-percentile among feasible portfolios, P(optimum),
  concentration ×uniform, %-exact-optimum. (Sign-safe; objectives can be negative.)
- **Q2 Language effect** — does the LLM view change the **exact** optimal portfolio? (Measured at the
  classical optimum, so QAOA noise does not confound it.)
- **Q3 Out-of-sample value** — realized test-window return of the view-tilted optimum minus the neutral
  optimum, with bootstrap CI + Wilcoxon, restricted to instances where the view actually changed the
  portfolio.

## Local validation (80 instances, n=12 assets, hold 4, p=3, CPU)
- **Q1:** QAOA finds the **exact** optimum in only **13.8%** of instances; its typical pick sits at the
  **71st percentile** of feasible portfolios; concentration **38× uniform**. → Vanilla QAOA at shallow
  depth is an *inconsistent* exact optimizer at this scale — an honest, literature-consistent result, not
  a polished success story.
- **Q2:** even a realistic ±10% LLM view changes the optimal portfolio in **98.8%** of instances
  (mean Jaccard 0.76). → Language has real leverage on the selection.
- **Q3 (PRELIMINARY, not yet valid):** −2.0%/quarter, 95% CI [−3.5%, −0.6%], Wilcoxon p=0.005, view beats
  neutral only 37% of the time.

## The honest caveat on Q3 (do not over-read)
The local validation applies **recent** headlines across **historical** rolling windows — an anachronism
(the news is not point-in-time at each decision date). So the Q3 magnitude is **not a valid backtest**; it
demonstrates the evaluation machinery only. The directionally-negative signal is *consistent* with the
finance literature that naive news/sentiment chasing tends to underperform, but we make **no claim** until
the valid test is run. **Q1 and Q2 are valid as reported.**

## Placebo / ablation control (`control.py`) — does the LANGUAGE matter, or would any tilt?
Because Q2/Q3 are classical (a tilt either does or doesn't change the exact optimum), we can run thousands
of instances locally and compare the real LLM tilt against matched **random** tilts (same sparsity +
magnitude, language removed), drawn 300×. On 1,600 instances (point-in-time news, n=12, 4 decision dates):
- **LLM OOS effect +1.29%**, sitting **outside the entire random-tilt null** (instance-level perm. p≈0).
  Random tilts that change the portfolio average +0.35%, so the *language-specific* premium is ~+0.9%.
- **BUT the honest unit is the decision date, not the instance** (subsets within a date are correlated).
  Per-date effect: 2023-12 **−5.5%**, 2024-03 +2.1%, 2024-06 +1.8%, 2024-09 +5.5% → **3/4 positive,
  MIXED**. The instance-level p is pseudo-replicated and overstated.
- **Verdict:** the LLM language carries signal beyond random noise and helps in 3 of 4 quarters, but it is
  **regime-dependent and NOT yet statistically robust** with only 4 periods. This is a pro-and-con finding,
  not a win — and it directly motivates more decision dates.

## Ablation: LLM vs RavenPack's own sentiment (ESS)
Same pipeline, but tilt by RavenPack's off-the-shelf **event_sentiment_score** instead of the LLM:
- **LLM** (reads headline text): changes 64% of optima, **+1.29%** OOS.
- **ESS** (averaged sentiment score): changes only **17%**, **−0.20%** OOS. → **LLM > ESS.**
The LLM extracts stronger, more actionable directional views from the text than the averaged score.
**Honest caveat:** ESS is window-averaged so its magnitude is smaller (hence it rarely crosses the
±10% tilt that flips the optimum); a magnitude-matched ESS would be a fairer test. As-is, the result is
"at equal tilt strength, language understanding beats the off-the-shelf sentiment number" — subject to
the same 4-period robustness caveat.

## What makes Q3 valid (the BigRed200 run)
Point-in-time news keyed to each decision date — **WRDS RavenPack** news analytics — plus the large
`(subset × window)` sweep at higher `n` on A100s. Submit `slurm/run_array.slurm`; aggregate with
`slurm/aggregate.py`. Only then do we report whether language-conditioning helps or hurts OOS, with the
full distribution.

## Reproduce locally
```bash
python -m paper.study.universe        # refresh the Dow-30 price snapshot (optional; committed)
python -m paper.study.news_views      # refresh real-headline LLM views (optional; committed)
python -m paper.study.run_study --n 12 --budget 4 --instances 16 --reps 3
python -m paper.study.figures
```
