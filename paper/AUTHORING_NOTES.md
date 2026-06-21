# Authoring notes — peer-review simulation, scope, and the human/machine split

This file is **not** part of the submission. It records the reasoning behind the poster abstract so the
author (Pradyot) can defend every claim and decide what to rewrite in his own voice.

---

## 1. Scope decision: two objectives, not three

We cut the IQAE/CVaR risk module from the poster. It is real and works, but (a) it is *replication* of
Woerner–Egger, not novelty; (b) it is risk estimation, not the language↔quantum bridge that makes this a
QNLP-relevant contribution; (c) three objectives on a poster reads as unfocused. The two we keep are
tightly coupled and tell one story:

- **O1 — Language-conditioned QAOA.** An LLM natural-language view becomes a tilt on μ that deforms the
  QAOA cost Hamiltonian and moves the ground state.
- **O2 — State-evolution visualization.** Snapshot the statevector per layer so the steering (and the
  optimizer's working process) is auditable.

The risk code stays in the repo (`circuits.py: iqae_cvar`, `experiments.py: fig_risk`) for the fuller
QLPC story / a future full paper — just not on this poster.

## 2. Is this the right topic and track? (honest answer)

**Topic — yes, and it is genuinely in a gap.** A literature check (June 2026) places us precisely between
two existing lines of work:

| Prior work | Views from | Optimizer | NL input? | State viz? |
|---|---|---|---|---|
| Lee et al. 2025 (`lee2025llmbl`) — LLM-Enhanced Black–Litterman | **LLM (language)** | **classical** | yes | no |
| Chen et al. 2023 (`chen2023qbl`) — Black–Litterman on NISQ | QML (numeric) | VQE (quantum) | no | no |
| Sun et al. 2025 (expert-analysis eval) | expert (as *metric*) | VQE | no | no |
| **This work** | **LLM (language)** | **QAOA (quantum)** | **yes** | **yes** |

So the precise, defensible novelty is: *routing an LLM's natural-language view into a QAOA cost
Hamiltonian and visualizing the resulting state evolution.* Neither neighbor does both. **Do not** claim
"first quantum portfolio with views" (false — Chen 2023) or "first LLM views in finance" (false — Lee
2025). The phrase used in the abstract — "first ... natural-language view visibly deforming the state
evolution of a quantum portfolio optimizer" — is the claim that survives.

**Track — Student Poster Session is the right fit** for a first submission: the contribution is a
focused, honest proof-of-concept with a strong *visual*, which is exactly what a poster rewards, and the
bar is calibrated for students. A 4-page short paper is the stretch option if you want the LNCS
publication; the work would need a reviewer to accept "exact simulation, n=4" as sufficient, which is
riskier.

## 3. Simulated peer review (3 reviewers)

**R1 (quantum optimization).** *"n=4, exact statevector — this is a toy; where is hardware or scaling?"*
→ Response: agreed, and stated up front in "Scope and honesty." The poster's claim is transparency + the
language bridge, **not** advantage. The instrument generalizes; the demo is intentionally small so the
state evolution is legible. *Action: keep the limitation prominent; do not overstate.*

**R2 (QNLP / NLP).** *"The 'LLM' is just a number added to μ — where is the language?"* → This is the
sharpest fair critique. In the current experiments the conviction vector is hand-set to emulate an LLM
view; we did not run a live LLM in the figure pipeline. *Action (author must decide):* either (i) soften
to "LLM-derived/representable conviction" and show one concrete worked example of a sentence → conviction
mapping on the poster, or (ii) add a small real LLM call (local Ollama is available in QLPC) that turns a
real headline into the conviction vector, and cite it. Option (ii) materially strengthens the QNLP claim
and is recommended if time permits before camera-ready.

**R3 (finance).** *"Additive tilt on μ is not Black–Litterman; don't oversell the finance."* → Agreed; we
say "in the spirit of" Black–Litterman, not that we implement its posterior. *Action: keep that hedge.*

Net: the work is poster-appropriate and honest. The one claim to firm up before camera-ready is R2 — make
the "language" part concrete (a real or clearly-illustrated sentence→view step).

## 4. Human vs machine authoring — recommendation

A reviewer (and your own integrity) needs the *ideas* to be yours. Split it this way:

**You (Pradyot) must own — rewrite in your own words and be able to defend live:**
- The novelty sentence and the literature-gap table (§2 above). Read the two key papers yourself first
  (links below) — do not cite from this summary alone.
- The interpretation of *why* language-steering matters and why visualization is the contribution.
- The limitations paragraph (you must agree with every hedge).
- The R2 decision: how you characterize the "LLM" step honestly.

**Assistant-generated, but you must verify (reproducible artifacts, lower authorship burden):**
- The circuit code, the math mapping (QUBO→Ising), the figures and numbers. These are checkable: run
  `python -m paper.experiments` and confirm `results.json` matches the abstract.

**Disclosure:** the abstract's acknowledgement already states AI assistance was used for implementation
and figures while the questions, positioning, interpretation, and claims are yours. Keep this — it is
honest and most venues now expect it. (Springer/COPE require that AI is *not* an author and that its use
is disclosed; this wording complies.)

Rule of thumb: if you cannot explain a sentence to R1/R2/R3 without re-reading this file, rewrite it until
you can. That is the line between grounded reasoning and machine generalization.

## 5. Read before submitting (do not skip)
- Lee et al. 2025, *LLM-Enhanced Black–Litterman Portfolio Optimization*, arXiv:2504.14345 (CIKM-W).
- Chen, Chung, Goan 2023, *Black–Litterman Portfolio Optimization with NISQ Computers*, arXiv:2312.00892.
- Farhi, Goldstone, Gutmann 2014, *QAOA*, arXiv:1411.4028.
- Barkoutsos et al. 2020, *Improving variational quantum optimization using CVaR*, Quantum 4:256.

## 6. Pre-submission checklist
- [ ] Read the four papers above; confirm the §2 gap table in your own words.
- [ ] Decide the R2 framing (illustrate or actually run the LLM→conviction step).
- [ ] Compile `abstract.tex` on Overleaf; confirm **≤ 2 pages**.
- [ ] Confirm `results.json` numbers match the abstract after a fresh `python -m paper.experiments`.
- [ ] Confirm Prof. Yuxi Hong is content to be acknowledged.
- [ ] Export `poster.pdf` (≤ 24"×36"); submit `abstract.pdf` + `poster.pdf` on OpenReview before Jun 21 AoE.
