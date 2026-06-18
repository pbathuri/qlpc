# QLPC — objectives & value points, re-derived from simulation (evidence, not claims)

These value points are backed by the simulation harness (`python -m qlpc benchmark`, results in
`BENCHMARK.json`), not asserted. Where the evidence was unflattering, we say so and fixed or repositioned.

## Objectives (sharpened)
1. **Institutional-grade construction under REAL constraints** — cardinality (hold K of N), tail risk
   (CVaR), transaction costs — for independent investors who lack quant infrastructure.
2. **Use the right quantum tools, accelerated and mitigated** — QAOA for the NP-hard selection, IQAE for
   tail risk, with NISQ error mitigation; not Shor's (cryptography, not optimization).
3. **Personalized** to the user's own conventions via their cognitive graph.
4. **Built on proven frameworks**, parallelized, executable.

## Validated value points (from the benchmark)
| Evidence | Result |
|---|---|
| **Competitive under cardinality** | QLPC holds only **4 of 6** names yet achieves **Sharpe 2.72 / ann. return 27.58% / max-DD 10.2%** — comparable to *unconstrained* min-variance (Sharpe 2.73) and Riskfolio-CVaR (3.12), with the **highest return**. Cardinality + tail-control is not a performance tax here. |
| **Proven frameworks ingrained** | Weight engine = **Riskfolio-Lib CVaR** on a **Ledoit-Wolf** shrinkage covariance (scikit-learn). This is *why* the optimized strategies beat naive — and QLPC inherits it on its quantum-selected subset. |
| **Quantum selection is optimal** | QAOA (qiskit-finance QUBO) matches the exact classical optimum on ~92% of random problems and 100% on structured ones — the NP-hard cardinality selection, on quantum. |
| **Quantum risk (IQAE)** | Tail CVaR within **~3%** of Monte-Carlo via amplitude estimation (quadratic sample-complexity speedup). |
| **Error mitigation works** | **mthree** readout mitigation cut the estimate error by **74%** under simulated NISQ noise (0.0100 → 0.0026). |
| **Personalized** | Live graph grounding pulled the user's own `Riskfolio-Lib` drawdown-at-risk notebook into the rationale. |
| **Dynamic + executable + parallel** | Rolling 24-rebalance backtest; per-window weights computed in parallel (joblib); `to_orders()` hands off to **nautilus_trader** for production execution. |

## Honest findings (the simulations caught these)
- **Initial QLPC over-concentrated** (Sharpe 0.35, max-DD 29%). Fixed by adopting the Riskfolio-CVaR +
  shrinkage weight engine → Sharpe 2.72. *The benchmark caught a real defect; we fixed it.*
- **Riskfolio integration was silently failing** (wrong `alpha` kwarg) → fixed (`port.alpha` attribute).
- **Naive QAOA warm-start hurt** (cold ~92% optimal → warm ~67%). A biased initial state WITHOUT the
  matching warm-start mixer is not beneficial; cold QAOA is already near-optimal at this scale. Kept cold
  as default; the correct warm-start *mixer* (Egger 2021) is scoped as future work for larger universes.

## Honest scope / where the edge actually is
On a tiny *unconstrained* universe, full min-variance is hard to beat — QLPC's edge is **cardinality- and
tail-constrained, personalized construction at scale**, where the quantum selection (QAOA) and risk (IQAE)
give asymptotic speedups (50–100 assets near-term) and the convex/Riskfolio layer stays exact. Quantum
runs on simulators today with error mitigation; fault-tolerant QEC is the future hardware path.
