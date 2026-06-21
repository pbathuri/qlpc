# Watching Language Steer a Quantum Optimizer: State-Evolution Visualization of LLM-Conditioned QAOA for Cardinality-Constrained Portfolio Selection

**Submission to QNLPAI 2026 (Quantum AI 2026), Indiana University Indianapolis.**
Track: *Quantum Computing for AI — Quantum optimization* (with a QNLP bridge via LLM-conditioning).

---

## Abstract

Variational quantum optimizers such as QAOA are usually evaluated by a single scalar — the
approximation ratio — which hides *how* the quantum state arrives at its answer and offers no place for
the qualitative, language-shaped views that drive real financial decisions. We present an instrumented,
fully state-resolved implementation of the quantum core of **QLPC**, a personalized portfolio
constructor, built on IBM Qiskit. Two contributions follow. First, we expose the **evolution of the
quantum state** through every QAOA layer for a cardinality-constrained Markowitz selection problem,
showing probability mass migrate from a uniform superposition onto the optimal feasible portfolio, and we
relate circuit depth to the sharpness of that concentration. Second, we introduce **language-conditioned
QAOA**: an LLM-extracted natural-language market view is encoded as a tilt on the expected-return vector,
which deforms the cost Hamiltonian and *visibly re-concentrates the quantum ground state* on a different
portfolio as conviction rises. We pair the optimizer with an Iterative Quantum Amplitude Estimation
(IQAE) circuit that reproduces the Woerner–Egger quantum risk-analysis primitive for tail / CVaR
estimation. All experiments run exactly on the statevector simulator with reproducible seeds. The result
is a transparent, auditable bridge between *language*, *quantum optimization*, and *finance* — and a
visualization methodology that makes a NISQ optimizer's working process legible.

---

## 1. Introduction

Quantum approaches to finance have concentrated on two hard kernels that classical convex solvers handle
poorly: **discrete, cardinality-constrained selection** (which *k* of *n* assets to hold), attacked with
the Quantum Approximate Optimization Algorithm (QAOA) [Farhi 2014]; and **tail-risk estimation**, where
amplitude estimation gives a quadratic sampling advantage over Monte-Carlo [Brassard 2002; Woerner &
Egger 2019; Egger et al. 2020]. Yet two gaps persist. (i) These optimizers are reported as black boxes:
a final bitstring and an approximation ratio, with no view of the *state evolution* that produced them —
unhelpful for a domain where "no room for error" demands an auditable process. (ii) Real allocation is
driven by **language** — analyst theses, filings, news — which has no natural port into a QUBO.

We address both with an instrumented re-implementation of the quantum core of **QLPC** (Quantum-LLM
Personalized portfolio Constructor). Our contributions:

1. **State-evolution visualization.** We snapshot the full statevector after the initial superposition
   and after every cost/mixer layer, rendering the probability distribution over all 2ⁿ candidate
   selections as it concentrates on the optimum. We quantify concentration as P(optimum) relative to a
   uniform guess, and trace its growth with circuit depth.
2. **Language-conditioned QAOA (the QNLP bridge).** An LLM emits a per-asset directional *conviction*
   vector from natural-language views; we add it to the expected returns, which reshapes the Ising cost
   Hamiltonian. We show that as linguistic conviction increases, the quantum ground state migrates
   continuously across portfolios — language *steering* a quantum optimizer.
3. **Reproducible risk replication.** An IQAE circuit loads a fat-tailed loss distribution into a qubit
   register and estimates the tail-exceedance probability, recovering the classical CVaR to within a few
   percent — a faithful replication of the canonical quantum risk-analysis result.

Everything is exact (statevector) on small, well-separated instances, with fixed seeds, so every figure
is reproducible from `paper/experiments.py`.

## 2. Method

### 2.1 Cardinality-constrained selection as an Ising Hamiltonian

We hold exactly *B* of *n* assets to maximize the Markowitz objective `f(x) = μ·x − q·xᵀΣx`, `x ∈ {0,1}ⁿ`
[Markowitz 1952]. We minimize the penalized cost
`C(x) = −μ·x + q·xᵀΣx + P·(Σx − B)²`,
and map `x_i = (1 − z_i)/2` to obtain a diagonal Ising Hamiltonian `H_C` over `I, Z_i, Z_iZ_j` terms. A
key practical point: the penalty `P` is set *just* above the largest marginal objective gain of adding or
removing one asset. Over-penalizing creates a near-degenerate feasible band, separated from infeasible
states by a large gap, that shallow QAOA cannot resolve — the optimizer spends its expressivity enforcing
feasibility and leaves the objective gradient invisible. A tight penalty keeps that gradient legible.

### 2.2 QAOA ansatz and state snapshots

We use the standard QAOA ansatz: prepare `|+⟩ⁿ`, then for each of *p* layers apply `exp(−iγ_k H_C)`
followed by the transverse-field mixer `exp(−iβ_k ΣX)`. Because `H_C` is diagonal, the Trotterized
`PauliEvolutionGate` is exact. Angles are optimized by COBYLA with multiple random restarts (the standard
defense against QAOA's non-convex angle landscape); the winning restart's evaluation trace is the
"working process" we visualize. After optimization we recompute the statevector incrementally, recording
`|ψ|²` after each layer.

### 2.3 Language conditioning

The LLM layer of QLPC reads natural-language views and emits a signed conviction `v_i ∈ [−1, 1]` per
asset. We form a tilt `μ → μ + s·v` (strength `s`), echoing the view-blending logic of Black–Litterman
[1992] but routed into the *quantum* cost Hamiltonian rather than a classical mean. Changing the language
changes `H_C`, hence the quantum ground state — the object of our steering experiment.

### 2.4 Quantum risk (IQAE)

For risk we follow Woerner & Egger [2019] and Grinko et al. [2021]: discretize the portfolio loss onto
*m* qubits, load `√pmf` with a state-preparation circuit, flag the tail above VaR with an
`IntegerComparator`, and run Iterative Quantum Amplitude Estimation to estimate the tail-exceedance
amplitude `P(loss ≥ VaR)`, from which CVaR follows. IQAE attains the `O(1/ε)` query scaling of amplitude
estimation versus `O(1/ε²)` for Monte-Carlo.

## 3. Results

*(All numbers are emitted by `paper/experiments.py` into `paper/results.json`; figures in `paper/figures/`.)*

### 3.1 The quantum state concentrates on the optimum (Fig 1, Fig 2)

On a well-separated 4-asset instance (`B = 2`), the QAOA output evolves from the uniform distribution
(P(optimum) = 1/16 = 0.062, across three layers **0.062 → 0.248 → 0.625**) to a sharply peaked
distribution on the optimal feasible selection {A, B}. At depth **p = 2** the optimum carries
**P(opt) = 0.63** — a **10.0× concentration** over a uniform guess — with an expected-objective
**approximation ratio of 0.92** and best-sample recovery of the exact optimum. Fig 1 shows the per-layer
probability heatmap (mass sweeping onto the starred optimal bitstring); Fig 2 shows approximation ratio
and P(optimum) rising with depth (AR 0.50 / 0.92 / 0.89 / 0.93 for p = 1…4; P(opt) up to 0.75 at p = 4),
with p = 1 insufficient and p = 2 already cracking the instance.

### 3.2 The working process is legible (Fig 3)

Fig 3 traces the COBYLA cost-expectation `⟨H_C⟩` descending to its optimum, and the single-qubit
entanglement entropy rising as the cost layers build correlations and then localizing as the state
concentrates — a qualitative readout of the optimizer "thinking."

### 3.3 Language steers the quantum ground state (Fig 4) — the QNLP result

Starting from the neutral optimum {A, B}, we feed an LLM bullish view on the two lower-return assets
{C, D}. As conviction rises the *exact* optimum migrates **{A,B} → {A,C} → {C,D}**, and the **QAOA modal
selection tracks the language-shifted optimum in all three regimes** (approximation ratios 0.92 / 0.92 /
0.85; Fig 4). The selection is thus a continuous function of language: weak conviction swaps a single
holding, strong conviction flips the book. Concentration is naturally lower under a strong view
(P(opt) ≈ 0.16–0.19) because the tilt creates near-ties among portfolios — an honest reflection of real
ambiguity, not optimizer failure. To our knowledge this is the first demonstration of a natural-language
view *visibly* deforming the state evolution of a quantum portfolio optimizer.

### 3.4 Risk replication (Fig 5)

The IQAE circuit loads a fat-tailed (Student-t) loss distribution onto 5 qubits and estimates the tail
probability at α = 0.95. The genuinely *quantum-estimated* quantity is the tail-exceedance probability
`P(loss ≥ VaR)`, which IQAE recovers to within **0.07%** of the loaded distribution's true value; the
CVaR then follows from that distribution. This reproduces the Woerner–Egger primitive on our instance.

## 4. Related work and replication

Our optimization core is QAOA [Farhi 2014]; the CVaR-as-objective variational idea is due to Barkoutsos
et al. [2020]; our risk circuit replicates Woerner & Egger [2019] with the IQAE estimator of Grinko et
al. [2021], in the lineage of quantum option pricing [Stamatopoulos et al. 2020] and the IBM quantum-
finance survey [Egger et al. 2020]. The QNLP context [Coecke et al. 2010; Lorenz et al. 2023] motivates
treating language as a first-class input to quantum computation; we differ by routing *LLM-derived* views
into a finance optimizer's Hamiltonian rather than encoding sentence grammar, and by foregrounding
*visualization* of the resulting state evolution.

## 5. Limitations and honesty

Results are on small (n = 4–8), well-separated instances simulated exactly; we make **no** speed or
quality claim against classical solvers at scale, and brute force remains the oracle we benchmark
against. QAOA's advantage on portfolio QUBOs at NISQ depth is unsettled; we report depth-dependent
concentration, not supremacy. The language tilt is a deliberately simple, interpretable mechanism
(additive on μ), not a learned encoder. The IQAE advantage is asymptotic in ε and not realized as wall-
clock speedup here. These constraints are stated so the contribution — *transparency and a language↔
quantum bridge* — is not overread.

## 6. Conclusion

By making a quantum optimizer's state evolution and working process fully visible, and by showing
natural-language views steer that evolution, we offer an auditable template for high-stakes quantum
finance and a concrete QNLP × quantum-optimization use case. Code and figures are reproducible from a
single script.

## Reproducibility

`python -m paper.experiments` regenerates every figure and `results.json` (Qiskit 2.x, Aer, fixed seeds).
The instrumented circuits live in `paper/circuits.py`.

## References

See `references.bib` (Farhi 2014; Brassard et al. 2002; Woerner & Egger 2019; Grinko et al. 2021;
Barkoutsos et al. 2020; Stamatopoulos et al. 2020; Egger et al. 2020; Markowitz 1952; Black & Litterman
1992; Rockafellar & Uryasev 2000; Coecke et al. 2010; Lorenz et al. 2023).
