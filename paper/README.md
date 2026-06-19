# QNLPAI 2026 paper — instrumented quantum circuits for QLPC

This folder contains a submission-ready short paper + a draft poster for **QNLPAI 2026** (Quantum AI
2026, Indiana University Indianapolis), built on the quantum core of **QLPC**. It re-implements QLPC's
QAOA portfolio selection and IQAE risk estimation on IBM Qiskit, *instrumented so the quantum state
evolution and the optimizer's working process are fully visible*, and adds **language-conditioned QAOA**
— an LLM natural-language view that visibly steers the quantum ground state.

Track fit: **Quantum Computing for AI → Quantum Optimization**, with a **QNLP** bridge (LLM views).
Deadline: **June 21, 2026 (AoE)** · Submission: **OpenReview**
(`https://openreview.net/group?id=qnlp.ai/QNLPAI/2026/Conference`).

## Files
| file | what |
|------|------|
| `circuits.py` | Instrumented QAOA (QUBO→Ising, per-layer statevector snapshots, language tilt) + IQAE CVaR |
| `experiments.py` | Generates every figure + `results.json` (deterministic, fixed seeds) |
| `poster.py` | Assembles `poster.pdf` / `poster.png` (A2) from the figures + headline numbers |
| `paper.tex` | Overleaf-ready short paper (numbers as macros at the top) |
| `paper.md` | Readable full draft of the same paper |
| `abstract_1page.md` | 1-page abstract for the student-poster track |
| `references.bib` | Bibliography |
| `figures/*.png` | fig1 state evolution · fig2 depth · fig3 process · fig4 language · fig5 risk |
| `results.json` | All numbers, for reproducibility |

## Reproduce
```bash
# from the qlpc repo root, using the qlpc venv (qiskit 2.x + aer + matplotlib + scipy)
python -m paper.experiments      # -> figures/*.png + results.json   (~1 min)
python -m paper.poster           # -> poster.pdf + poster.png
```

## Compiling the paper
No LaTeX engine is installed locally. Compile `paper.tex` + `references.bib` on **Overleaf**
(New Project → Upload, add the `figures/` folder), or install MiKTeX/TeX Live and run
`pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`.

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
- [ ] Compile `paper.tex` → `paper.pdf` (Overleaf).
- [ ] Confirm the chosen track on OpenReview (full paper vs. 1-page abstract + poster).
- [ ] Add author name/affiliation in `paper.tex` (currently anonymous placeholder).
- [ ] Upload `paper.pdf` (and `poster.pdf` if submitting to the poster track) before Jun 21 AoE.
