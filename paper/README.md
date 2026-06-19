# QNLPAI 2026 paper — instrumented quantum circuits for QLPC

This folder contains the **QNLPAI 2026** Student-Poster-Session submission (2-page LNCS abstract + draft
poster) built on the quantum core of **QLPC**, plus a longer short-paper draft for later. It re-implements
QLPC's QAOA portfolio selection on IBM Qiskit, *instrumented so the quantum state evolution is fully
visible*, and adds **language-conditioned QAOA** — an LLM natural-language view that visibly steers the
quantum ground state.

**Target track: Student Poster Session.** Submit `abstract.pdf` (from `abstract.tex`) + `poster.pdf`.

**Scope — exactly two objectives** (see `AUTHORING_NOTES.md`):
- **O1** Language-conditioned QAOA: an LLM view tilts μ, deforming the cost Hamiltonian.
- **O2** State-evolution visualization: per-layer statevector snapshots make the steering auditable.

The IQAE/CVaR **risk module is out of scope for the poster** (it is replication, not the language bridge);
it stays in the repo (`circuits.py:iqae_cvar`, `experiments.py:fig_risk`) for a future full paper.

Track fit: **Quantum Computing for AI → Quantum Optimization**, with a **QNLP** bridge (LLM views).
Deadline: **June 21, 2026 (AoE)** · Submission: **OpenReview**
(`https://openreview.net/group?id=qnlp.ai/QNLPAI/2026/Conference`).

### Format requirements (from the CFP)
- **Short paper: 4 pages max** (LNCS), excluding references and an optional appendix.
  (Full papers are 8 pages — we target the short-paper track.)
- **Template: Springer Nature LNCS** (`llncs.cls` + `splncs04.bst`), via the Overleaf LNCS template or
  the Word template.
- **Student Poster Session:** a **2-page** LNCS abstract **+** a draft poster PDF.
- **Poster size:** ≤ **24" wide × 36" high** (portrait). `poster.pdf` is generated at exactly 24×36.

## Files
| file | what |
|------|------|
| `circuits.py` | Instrumented QAOA (QUBO→Ising, per-layer statevector snapshots, language tilt) + IQAE CVaR |
| `experiments.py` | Generates every figure + `results.json` (deterministic, fixed seeds) |
| `poster.py` | Assembles `poster.pdf` / `poster.png` at **24"×36"** from the figures + headline numbers |
| `paper.tex` | **LNCS** short paper — 4-page body + appendix (longer story, all 3 modules) |
| `abstract.tex` | **LNCS** 2-page poster-track abstract — **the submission** (2 objectives, lit-positioned) |
| `AUTHORING_NOTES.md` | Peer-review simulation, scope rationale, human/machine split, reading list (not submitted) |
| `paper.md` | Readable full draft of the paper (Markdown mirror) |
| `abstract_1page.md` | Plain-text condensed abstract (reference; the LNCS one is `abstract.tex`) |
| `references.bib` | Bibliography (LNCS `splncs04` style) |
| `figures/*.png` | fig1 state evolution · fig2 depth · fig3 process · fig4 language · fig5 risk |
| `results.json` | All numbers, for reproducibility |

## Reproduce
```bash
# from the qlpc repo root, using the qlpc venv (qiskit 2.x + aer + matplotlib + scipy)
python -m paper.experiments      # -> figures/*.png + results.json   (~1 min)
python -m paper.poster           # -> poster.pdf + poster.png (24x36)
```

## Compiling the paper (no local LaTeX engine)
Use the **Overleaf "Springer Nature LNCS" template** (it provides `llncs.cls` + `splncs04.bst`):
1. New Project → from the LNCS template (or upload `paper.tex`, `abstract.tex`, `references.bib`).
2. Upload the `figures/` folder.
3. Compile. `paper.tex` and `abstract.tex` already use `\documentclass{llncs}`.

Keep the **body ≤ 4 pages**; the three supporting figures are in the appendix (allowed as extra pages).
Check the compiled length before submitting and trim prose if the body spills past page 4.

## Headline results (see `results.json`)
- **State evolution (p=2):** P(optimum) 0.062 → 0.625 (10.0× a uniform guess), approx-ratio 0.92, exact
  optimum recovered; p=4 reaches P(opt)=0.75.
- **Language steering:** a graded LLM bullish view migrates the optimum {A,B} → {A,C} → {C,D}; the QAOA
  modal selection tracks it in all three regimes.
- **IQAE risk:** tail-exceedance probability recovered to 0.07% of the loaded distribution.

## Honest scope
Small (n=4–8), well-separated, exactly-simulated instances. No scaling/supremacy claim — brute force is
the oracle. The language tilt is a simple additive mechanism; the IQAE advantage is asymptotic in ε.
The contribution is **transparency + a language↔quantum↔finance bridge**, not a performance record.

## Submission checklist (you do these — I can't submit for you)
- [ ] Create an **OpenReview profile** (the CFP asks for this ≥2 weeks ahead — do it now).
- [ ] Fill in author name / affiliation / ORCID / email in `paper.tex` (and `abstract.tex`) — currently
      placeholders (`First Author`, `author@example.com`).
- [ ] Compile on Overleaf with the **LNCS** template → `paper.pdf`; **verify the body is ≤ 4 pages**.
- [ ] Pick the track: **short paper** (`paper.pdf`) for the main conference, **or** the **poster
      session** (`abstract.pdf` from `abstract.tex`, ≤ 2 pages, **plus** `poster.pdf`).
- [ ] Upload the PDF(s) on OpenReview before **June 21, 2026 (AoE)**.
