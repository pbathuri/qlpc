# Watching Language Steer a Quantum Optimizer
### State-Evolution Visualization of LLM-Conditioned QAOA for Cardinality-Constrained Portfolio Selection

**QNLPAI 2026 — Student Poster Track.** Track fit: *Quantum Computing for AI — Quantum Optimization*, with a QNLP bridge.

**Problem.** Variational quantum optimizers (QAOA) are reported as black boxes — a final bitstring and a
single approximation ratio — which hides *how* the quantum state reaches its answer and offers no place
for the qualitative, language-shaped views that drive real financial decisions. In high-stakes
quant workflows ("no room for error"), an opaque optimizer is unusable.

**Approach.** We re-implement the quantum core of **QLPC** (a Quantum–LLM Personalized portfolio
Constructor) on IBM Qiskit, instrumented so every intermediate quantum state is observable:
1. **State-evolution visualization.** Cardinality-constrained Markowitz selection is mapped to a diagonal
   Ising Hamiltonian and solved with QAOA. We snapshot the full statevector after the initial
   superposition and after every cost/mixer layer, rendering the probability over all 2ⁿ candidate
   portfolios as it concentrates onto the optimum.
2. **Language-conditioned QAOA (QNLP bridge).** An LLM emits a signed per-asset *conviction* vector from
   natural-language market views; we add it to expected returns, deforming the cost Hamiltonian. As
   conviction rises, the quantum ground state migrates continuously across portfolios — language
   *steering* a quantum optimizer.
3. **Quantum risk (IQAE).** An Iterative Quantum Amplitude Estimation circuit loads a fat-tailed loss
   distribution and estimates the tail / CVaR, replicating the Woerner–Egger primitive.

**Results (exact statevector simulation, fixed seeds).**
- On a well-separated 4-asset instance (hold 2), QAOA at depth *p=2* drives **P(optimum) from 1/16 =
  0.062 to 0.63** (layers: 0.062 → 0.248 → 0.625) — a **10.0× concentration** over a uniform guess —
  with approximation ratio **0.92** and best-sample recovery of the exact optimum; p=4 reaches P(opt)=0.75.
- A graded LLM bullish view on the two lower-return assets migrates the optimum **{A,B} → {A,C} → {C,D}**,
  and the **QAOA modal selection tracks the shifted optimum in all three regimes** — the selection is a
  continuous function of language.
- The IQAE risk circuit recovers the tail-exceedance probability `P(loss ≥ VaR)` to within **0.07%** of
  the loaded distribution (the genuinely quantum-estimated quantity); CVaR follows from that distribution.

**Contribution.** To our knowledge the first demonstration of a natural-language view *visibly* deforming
the state evolution of a quantum portfolio optimizer — an auditable, reproducible bridge between
language, quantum optimization, and finance.

**Limitations (stated up front).** Small, well-separated, exactly-simulated instances; no scaling or
supremacy claim (brute force is the oracle); additive language tilt; asymptotic (not wall-clock) IQAE
advantage.

**Reproducibility.** `python -m paper.experiments` regenerates every figure and `results.json`
(Qiskit 2.x + Aer). Code: `paper/circuits.py`, `paper/experiments.py`.
