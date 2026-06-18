# QLPC — Quantum-LLM Personalized portfolio Constructor

**IEEE Grandtech 26 · FinTech route.** A hybrid **quantum + classical + LLM** system for *dynamic,
personalized* portfolio construction. It fuses the right quantum tools (not the wrong ones), exact convex
optimization, an LLM reasoning loop, and grounding in the user's **own cognitive graph** — and it runs
end-to-end today on simulators, degrading to faithful classical equivalents so it never breaks.

```
brief (natural language)
   │  ground in YOUR graph (PCE/Cartograph)  ──► LLM elicits constraints (budget, risk, CVaR α)
   ▼
market data ─► [QUANTUM] QAOA selection (which K of N — NP-hard QUBO) ─► [CLASSICAL] convex mean-CVaR
                                                                              weights (CVXPY, exact)
                        [QUANTUM] IQAE tail risk (CVaR via amplitude estimation) ─┘
   ▼
constructed portfolio + quantum CVaR + LLM rationale grounded in your work  ─► (backtest/execute: nautilus_trader)
```

## Why these quantum algorithms (the honest choice)
- **Shor's algorithm is the wrong tool** — it factors integers (cryptography), not portfolios. We don't use
  it as an optimizer (only relevant to FinTech as the post-quantum *threat*).
- **Iterative Quantum Amplitude Estimation (IQAE)** → tail risk. Estimates the exceedance probability
  `P(loss ≥ VaR)` with a **quadratic** sample-complexity speedup over Monte-Carlo (Woerner–Egger 2019).
  This is the *real* version of the classical-CVaR placeholder in your `captain-whiskers` optimizer.
- **QAOA** → the NP-hard, cardinality-constrained **asset selection** (QUBO/Ising) — the discrete piece
  classical convex solvers can't do. Same gate-based, NISQ-suitable family as your `captain-whiskers` VQE.
- **Classical convex (CVXPY)** → exact continuous mean-CVaR weights on the selected subset. Don't quantize
  what a convex solver already solves optimally.

## What makes it different
Quantum-finance demos do *one* isolated piece on toy data; robo-advisors are generic. QLPC fuses hybrid
quantum (selection **and** risk) + convex optimization + an LLM reasoning loop **personalized by your own
published graph engine** (Cartograph / PCE) — end-to-end, explainable, dynamically rebalancing, tailored
to *you*.

## Run it
```bash
python -m venv .venv && .venv/Scripts/activate     # (already set up here)
pip install -r requirements.txt                     # numpy scipy cvxpy qiskit qiskit-finance ...
python -m qlpc demo                                 # full pipeline on a synthetic market
python -m qlpc construct "conservative, ~4 names, tail-aware" --tickers AAPL MSFT NVDA JPM XOM JNJ --live
pytest -q                                           # tests (pass with or without qiskit)
```
Every quantum module prints which path ran (`quantum-qaoa` / `quantum-iqae` vs the classical fallback) and
the classical cross-check, so results are always honest and verifiable.

## Modules
| File | Role |
|---|---|
| `data.py` | market data (yfinance / synthetic) → μ, Σ |
| `quantum_select.py` | **QAOA** cardinality-constrained selection (QUBO) + exact classical oracle |
| `quantum_risk.py` | **IQAE** tail-risk CVaR + classical-MC oracle |
| `convex.py` | CVXPY mean-variance / mean-CVaR exact weights |
| `risk.py` | stochastic scenarios + classical VaR/CVaR |
| `llm.py` | constraint elicitation + grounded explanation (local/cloud, rule-based fallback) |
| `graph.py` | personalization via your PCE/Cartograph memory graph |
| `pipeline.py` / `cli.py` | orchestration + CLI |

## Status
**Prototype — runnable proof of concept.** Built on shipped components (Cartograph v1.0.1 graph engine,
your `captain-whiskers` quantum optimizer + `AutoResearchClaw` qiskit skill). Quantum paths run on
`qiskit-aer` simulators; scales to ~50–100 assets on near-term hardware.
