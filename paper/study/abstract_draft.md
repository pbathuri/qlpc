# DRAFT — 2-page LNCS poster abstract (honest, pros-and-cons)

Working draft to react to. Numbers marked `[HPC]` get finalized from the BigRed200 `study_results.json`;
everything else uses the validated local + control numbers and will change little. I'll convert this to
`llncs` `.tex` once the HPC run lands. Tone follows your standard: report what works AND what doesn't, and
say plainly how it builds on prior work and what it offers the community.

---

## Title
**Does Language Help a Quantum Portfolio Optimizer? A Bias-Controlled, Point-in-Time Study of
LLM-Conditioned QAOA**

*Pradyot Bathuri — Indiana University, Bloomington. QNLPAI 2026, Student Poster Session.*

## Abstract (~120 words)
Two strands of recent work sit on either side of a gap: LLM-conditioned Black–Litterman allocation
(classical; Lee et al. 2025) and quantum portfolio optimization with *numerically* predicted views (VQE,
no language; Chen et al. 2023). We ask the question between them — *can a natural-language view, read by an
LLM, usefully steer a quantum optimizer, and can we trust the result?* — and we answer it with a
deliberately bias-controlled, out-of-sample study: an objective universe (Dow 30 / CRSP), point-in-time
news (WRDS RavenPack) read by a local LLM into per-asset convictions, language-conditioned QAOA solved
exactly, and pre-registered metrics over hundreds of instances on HPC (BigRed200 A100s). The honest
finding is mixed and that is the point: language carries signal beyond random tilts, but it is
regime-dependent and not yet robust. We contribute the transparent, reproducible pipeline and a sober
evaluation, not a performance claim.

## 1. Motivation and position in the literature
Quantum portfolio work maps asset *selection* to a QUBO solved by QAOA [Farhi 2014] or VQE, and is usually
judged by one approximation ratio — opaque, and with no port for the *language* (theses, filings, news)
that drives real allocation. Two recent lines touch each side but not the middle: **Lee et al. (2025)**
drive Black–Litterman views with an LLM but optimize **classically**; **Chen et al. (2023)** bring views
into a **quantum** pipeline, but the views are **numerically predicted by QML** and solved with VQE — no
natural language, no view of the quantum state. We occupy the gap: an LLM reads real news → conviction →
a tilt on the QAOA cost Hamiltonian, evaluated with a rigor designed to *resist* favorable-result bias.

## 2. Method (the architected, bias-controlled pipeline)
- **Objective universe**, not author-picked: Dow 30 constituents; gold-standard **CRSP** total returns
  via WRDS. Ledoit–Wolf covariance.
- **Point-in-time language**, not our words: **WRDS RavenPack** headlines timestamped *before* each
  decision date, read by a local LLM (Ollama, qwen2.5-14B) into signed convictions — no look-ahead.
- **Language-conditioned QAOA**: conviction → a ±10% annual return-view tilt (the sentiment→view
  convention, cf. Riskfolio-Lib), deforming a diagonal Ising cost Hamiltonian; per-layer statevector
  snapshots. **Device-agnostic** fast simulator (numpy/CPU, cupy/**A100 on BigRed200**); the
  many-instance sweep runs as a SLURM array.
- **Pre-registered metrics, distributions not anecdotes**, over hundreds of (subset × rolling-window)
  instances with strict train/test separation: **Q1** optimizer quality, **Q2** language effect at the
  *exact* optimum, **Q3** out-of-sample value — plus a **placebo control** (matched *random* tilts) that
  asks whether the language matters or any tilt would do.

## 3. Results — what works, and what doesn't
- **Q1 (optimizer, honest):** shallow QAOA is an *inconsistent* exact optimizer — it recovers the true
  optimum in only ~14–27% of instances (median ~71st–94th percentile; ~40–66× concentration over uniform)
  `[HPC: refine at n=14, p=4 over ~1,600 instances]`. Useful as a sampler, not a solver, at this depth.
- **Q2 (language has leverage):** a realistic ±10% LLM view changes the *exact* optimal portfolio in ~99%
  of instances. The language moves decisions.
- **Ablation (LLM vs off-the-shelf sentiment):** tilting by RavenPack's own averaged event-sentiment
  score changes only ~17% of optima (−0.2% OOS) vs the LLM's 64% (+1.3%) — at equal tilt strength, the
  LLM reading headline *text* extracts more actionable signal than the sentiment number. (Caveat: ESS is
  window-averaged and thus weaker in magnitude; a magnitude-matched test is future work.)
- **Q3 / placebo (the nuanced core):** the LLM tilt's out-of-sample effect (+1.29%) sits **outside the
  entire distribution of 300 matched random tilts** (which average +0.35%) — so ~+0.9% is
  *language-specific*, beyond a generic tilt effect. **But** clustered by the honest unit — the decision
  date — it is **mixed**: −5.5%, +2.1%, +1.8%, +5.5% across four quarters (3/4 positive). With only four
  periods the instance-level significance is pseudo-replicated and overstated. **Verdict: promising,
  regime-dependent, not yet robust.** `[HPC: the BigRed200 run adds instances and — with a longer
  CRSP+RavenPack pull — more decision dates, which is what would settle significance.]`

## 4. Contribution, limitations, and reproducibility
**Contribution to the community:** (i) a transparent, end-to-end, reproducible bridge from *language* to a
*quantum* optimizer in finance, filling the Lee–Chen gap; (ii) a bias-control template (objective
universe, point-in-time news, pre-registered metrics, placebo control) that we think the QNLP×finance
area needs more of; (iii) an honest negative-leaning result — naive LLM-sentiment conditioning is *not*
reliably profitable — which is more useful to the field than another favorable anecdote.
**Limitations (stated plainly):** few decision dates; small *n* (exact simulation); a simple additive
tilt; no transaction costs; shallow QAOA. **Reproducibility:** one command (`python -m paper.study.…`),
fixed seeds, committed CRSP snapshot + point-in-time conviction cache; HPC via SLURM.

**Acknowledgements.** The author thanks Professor Yuxi Hong for guidance. Implementation and figures used
an AI coding assistant; the questions, positioning, interpretation, and claims are the author's own, and
all results are reproducible from released code.

---

### Figures to include (have PDFs already)
1. `study_q1_quality.pdf` — QAOA rank-percentile + concentration distributions.
2. `control_placebo.pdf` — LLM vs random-tilt null **and** per-decision-date effect (the honest unit).
*(Optional 3rd: a state-evolution snapshot for the visual hook.)*

### Open question for you
The strongest single framing is the **placebo + per-date** result — it's novel, honest, and memorable.
Do you want the poster to lead with **"does language carry signal beyond noise?"** (the control) or with
**"a transparent language→quantum pipeline"** (the method)? My vote: lead with the control finding, use
the pipeline as the how.
