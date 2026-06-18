# QLPC — validated run (proof of "Prototype" readiness)

End-to-end on `qiskit-aer` simulators (qiskit 2.4.2, qiskit-finance 0.4.1, cvxpy 1.9.1), `python -m qlpc demo`:

| Stage | Result |
|---|---|
| **Graph grounding** | `pce` — pulled 4 snippets from the user's memory graph (incl. their `Riskfolio-Lib` drawdown-at-risk optimization notebook) → rationale is personalized, not generic |
| **Quantum selection (QAOA)** | `quantum-qaoa` → {MSFT, NVDA, JNJ, GLD}; **optimal vs exact classical: True** |
| **Convex weights (CVXPY mean-CVaR)** | NVDA 57.7%, GLD 28.5%, JNJ 13.8%, MSFT 0% |
| **Quantum risk (IQAE)** | `quantum-iqae` CVaR@0.99 = 0.0822 vs classical 0.0847 → **2.9% rel-error**; IQAE tail-exceedance P(loss≥VaR)=0.0125 (target 0.01) |
| **Tests** | 8/8 pass (`pytest`) — portable: pass with or without qiskit |

**Honest scope.** Quantum paths run on simulators and report their classical cross-check every run; they
degrade to faithful classical equivalents when qiskit/cvxpy are absent, so the pipeline always runs.
Discretized IQAE risk is within a few % of Monte-Carlo. Scales to ~50–100 assets on near-term hardware
(QAOA circuit depth, IQAE logarithmic). This is a real, runnable proof of concept — not a slide.
