"""Instrumented IBM-Qiskit circuits for QLPC — built to EXPOSE the quantum state evolution.

This is the paper-facing re-implementation of QLPC's two quantum primitives, rewritten so every
intermediate quantum state is observable:

  * `LanguageConditionedQAOA` — cardinality-constrained portfolio SELECTION as a QUBO -> Ising
    Hamiltonian, solved by QAOA (Farhi 2014). An LLM "view" (natural-language market opinion) is
    encoded as a tilt on the expected-return vector mu, which deforms the cost Hamiltonian H_C. We
    snapshot the statevector after the initial superposition and after every cost/mixer layer, so the
    evolution of the selection distribution is fully visible.

  * `iqae_cvar_circuit` — tail / CVaR risk via Iterative Quantum Amplitude Estimation (Brassard 2002;
    Woerner & Egger 2019): load the loss distribution into n qubits, flag the tail with an
    IntegerComparator, estimate the exceedance amplitude.

Everything runs on the statevector simulator (small instances, exact). No external service.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace, entropy


# --------------------------------------------------------------------------------------------------
# QUBO  ->  Ising Hamiltonian
# --------------------------------------------------------------------------------------------------
def portfolio_qubo(mu: np.ndarray, sigma: np.ndarray, budget: int, q: float,
                   penalty: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Cardinality-constrained Markowitz as a QUBO. Minimise
        C(x) = -mu·x + q·xᵀΣx + P·(Σx - B)²,   x ∈ {0,1}ⁿ
    Returns (linear a, quadratic b upper-triangular, const)."""
    n = len(mu)
    if penalty is None:
        # Just enough to make the budget binding: a touch above the largest marginal objective gain of
        # adding/removing one asset. Bigger than this over-penalises, creating a near-degenerate feasible
        # band that shallow QAOA cannot resolve; this keeps the objective gradient visible to the optimiser.
        penalty = float(np.abs(mu).max() + q * np.abs(sigma).sum(axis=1).max() + 0.25)
    a = np.zeros(n)
    b = np.zeros((n, n))
    const = penalty * budget ** 2
    for i in range(n):
        a[i] += -mu[i] + q * sigma[i, i] + penalty * (1 - 2 * budget)
        for j in range(i + 1, n):
            b[i, j] += 2 * q * sigma[i, j] + 2 * penalty
    return a, b, const


def qubo_to_ising(a: np.ndarray, b: np.ndarray, const: float) -> tuple[SparsePauliOp, float]:
    """Map a QUBO (x∈{0,1}) to an Ising Hamiltonian (z∈{+1,-1}, x=(1-z)/2). Returns (H, offset)."""
    n = len(a)
    z = np.zeros(n)
    zz: dict[tuple[int, int], float] = {}
    offset = const
    for i in range(n):
        offset += a[i] / 2.0
        z[i] += -a[i] / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            if b[i, j] == 0:
                continue
            offset += b[i, j] / 4.0
            z[i] += -b[i, j] / 4.0
            z[j] += -b[i, j] / 4.0
            zz[(i, j)] = zz.get((i, j), 0.0) + b[i, j] / 4.0
    terms = []
    for i in range(n):
        if abs(z[i]) > 1e-12:
            terms.append(("Z", [i], z[i]))
    for (i, j), c in zz.items():
        if abs(c) > 1e-12:
            terms.append(("ZZ", [i, j], c))
    H = SparsePauliOp.from_sparse_list(terms or [("I" * 0, [], 0.0)], num_qubits=n)
    return H, offset


def objective(x: np.ndarray, mu, sigma, q) -> float:
    """The thing we actually care about (higher = better): mu·x - q·xᵀΣx."""
    return float(mu @ x - q * x @ sigma @ x)


def classical_best(mu, sigma, budget, q) -> tuple[list[int], float]:
    """Exact best feasible subset — the oracle for approximation-ratio reporting."""
    n = len(mu)
    best, best_o = None, -np.inf
    for c in itertools.combinations(range(n), budget):
        x = np.zeros(n); x[list(c)] = 1
        o = objective(x, mu, sigma, q)
        if o > best_o:
            best, best_o = list(c), o
    return best, best_o


# --------------------------------------------------------------------------------------------------
# Language → expected-return tilt  (the QNLP bridge)
# --------------------------------------------------------------------------------------------------
def view_tilt(n: int, views: dict[int, float], strength: float = 1.0) -> np.ndarray:
    """Turn LLM-extracted directional views {asset_idx: signed_conviction in [-1,1]} into a mu tilt.

    In QLPC the LLM reads news/filings and emits per-asset convictions; here that vector is added to
    the expected returns, shifting the QUBO ground state. This is the lever by which *language* steers
    the quantum optimiser — the object of the state-evolution study."""
    t = np.zeros(n)
    for i, v in views.items():
        t[i] = strength * float(v)
    return t


# --------------------------------------------------------------------------------------------------
# QAOA ansatz with per-layer statevector snapshots
# --------------------------------------------------------------------------------------------------
def qaoa_circuit(H: SparsePauliOp, gammas, betas) -> QuantumCircuit:
    """Standard QAOA: |+>^n, then for each layer exp(-i γ H_C) followed by the X-mixer exp(-i β ΣX).
    H_C is diagonal (I/Z/ZZ) so the Trotterised PauliEvolutionGate is EXACT."""
    n = H.num_qubits
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        qc.append(PauliEvolutionGate(H, time=float(gamma)), range(n))
        for qb in range(n):
            qc.rx(2.0 * float(beta), qb)
    return qc


def layer_states(H: SparsePauliOp, gammas, betas) -> list[Statevector]:
    """Statevector after the initial superposition and after each (cost, mixer) layer — the evolution."""
    n = H.num_qubits
    snaps = []
    qc = QuantumCircuit(n)
    qc.h(range(n))
    snaps.append(Statevector(qc))                         # layer 0: uniform superposition
    for gamma, beta in zip(gammas, betas):
        qc.append(PauliEvolutionGate(H, time=float(gamma)), range(n))
        for qb in range(n):
            qc.rx(2.0 * float(beta), qb)
        snaps.append(Statevector(qc))
    return snaps


def expectation(H: SparsePauliOp, sv: Statevector) -> float:
    return float(np.real(sv.expectation_value(H)))


def entanglement_entropy(sv: Statevector, n: int) -> float:
    """Von-Neumann entropy of the first qubit's reduced state — a scalar 'how entangled' readout."""
    rho = partial_trace(sv, list(range(1, n)))
    return float(np.real(entropy(rho, base=2)))


@dataclass
class QAOARun:
    H: SparsePauliOp
    offset: float
    mu: np.ndarray
    sigma: np.ndarray
    budget: int
    q: float
    reps: int
    gammas: np.ndarray = field(default=None)
    betas: np.ndarray = field(default=None)
    energy_history: list = field(default_factory=list)   # ⟨H_C⟩ per optimizer evaluation
    snapshots: list = field(default_factory=list)         # probability arrays per layer
    best_bitstring: str = ""
    selected: list = field(default_factory=list)
    qaoa_objective: float = 0.0                            # EXPECTED objective over the feasible output
    p_opt: float = 0.0                                     # probability mass on the exact optimum
    p_uniform: float = 0.0                                 # 1/2ⁿ baseline, for the concentration story
    best_sampled_optimal: bool = False                    # would best-of-samples recover the optimum?
    p_opt_layers: list = field(default_factory=list)       # P(optimum) after each layer — the evolution
    classical_selected: list = field(default_factory=list)
    classical_objective: float = 0.0

    @property
    def approx_ratio(self) -> float:
        """Expected quality of a QAOA sample relative to the exact optimum (feasible-projected)."""
        if self.classical_objective == 0:
            return 1.0
        return self.qaoa_objective / self.classical_objective

    @property
    def concentration(self) -> float:
        """How much more likely the optimum is than a uniform guess (×)."""
        return self.p_opt / self.p_uniform if self.p_uniform else 0.0


class LanguageConditionedQAOA:
    """QAOA portfolio selection whose cost Hamiltonian is shaped by an LLM language view on mu."""

    def __init__(self, mu, sigma, budget, q=0.5, reps=2, penalty=None):
        self.mu0 = np.asarray(mu, float)
        self.sigma = np.asarray(sigma, float)
        self.budget = int(budget)
        self.q = float(q)
        self.reps = int(reps)
        self.penalty = penalty

    def build(self, views: dict | None = None, strength: float = 1.0):
        mu = self.mu0 + (view_tilt(len(self.mu0), views, strength) if views else 0.0)
        a, b, const = portfolio_qubo(mu, self.sigma, self.budget, self.q, self.penalty)
        H, offset = qubo_to_ising(a, b, const)
        return mu, H, offset

    def optimize(self, views: dict | None = None, strength: float = 1.0,
                 maxiter: int = 200, seed: int = 7, restarts: int = 8) -> QAOARun:
        """Optimise the QAOA angles with `restarts` random COBYLA starts (best ⟨H_C⟩ wins) — the
        standard defence against QAOA's non-convex angle landscape. `energy_history` keeps the
        evaluation trace of the WINNING restart, i.e. the working process we visualise."""
        from scipy.optimize import minimize
        mu, H, offset = self.build(views, strength)
        n = len(mu)
        run = QAOARun(H=H, offset=offset, mu=mu, sigma=self.sigma, budget=self.budget,
                      q=self.q, reps=self.reps)
        run.classical_selected, run.classical_objective = classical_best(mu, self.sigma, self.budget, self.q)

        rng = np.random.default_rng(seed)
        best_e, best_x, best_hist = np.inf, None, []
        for _ in range(max(1, restarts)):
            hist: list = []

            def energy(theta, _h=hist):
                e = expectation(H, Statevector(qaoa_circuit(H, theta[:self.reps], theta[self.reps:])))
                _h.append(e)
                return e

            x0 = rng.uniform(0.0, 2 * np.pi, size=2 * self.reps)
            res = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter})
            if res.fun < best_e:
                best_e, best_x, best_hist = res.fun, res.x, hist
        run.gammas, run.betas = best_x[:self.reps], best_x[self.reps:]
        run.energy_history = best_hist

        # state evolution at the optimised parameters
        snaps = layer_states(H, run.gammas, run.betas)
        run.snapshots = [np.abs(sv.data) ** 2 for sv in snaps]

        # index -> bits (qiskit little-endian: qubit k contributes bit k); precompute feasibility + objective
        dim = 2 ** n
        bits_of = [[(idx >> k) & 1 for k in range(n)] for idx in range(dim)]
        feasible = np.array([sum(b) == self.budget for b in bits_of])
        objs = np.array([objective(np.array(b, float), mu, self.sigma, self.q) for b in bits_of])
        opt_idx = int(np.where(feasible)[0][np.argmax(objs[feasible])])   # exact optimum basis index

        run.p_uniform = 1.0 / dim
        run.p_opt_layers = [float(p[opt_idx]) for p in run.snapshots]     # the evolution of P(optimum)
        run.p_opt = run.p_opt_layers[-1]

        final = run.snapshots[-1]
        pf = final * feasible
        pf = pf / pf.sum() if pf.sum() > 0 else pf
        run.qaoa_objective = float((pf * objs).sum())                    # expected feasible objective
        bidx = int(np.argmax(final * feasible))                          # most-probable feasible sample
        run.selected = [i for i in range(n) if bits_of[bidx][i] == 1]
        run.best_bitstring = "".join(str(b) for b in reversed(bits_of[bidx]))
        run.best_sampled_optimal = (bidx == opt_idx)
        return run


# --------------------------------------------------------------------------------------------------
# IQAE tail / CVaR risk circuit
# --------------------------------------------------------------------------------------------------
def discretise_losses(losses: np.ndarray, n_qubits: int):
    lo, hi = float(losses.min()), float(losses.max())
    if hi <= lo:
        hi = lo + 1e-6
    bins = 2 ** n_qubits
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(losses, bins=edges)
    pmf = counts / counts.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pmf, centers, lo, hi


def iqae_cvar(losses: np.ndarray, alpha: float = 0.95, n_qubits: int = 5,
              epsilon: float = 0.01) -> dict:
    """Tail-exceedance probability P(loss ≥ VaR) by IQAE, then CVaR over the loaded distribution.
    Returns the quantum estimate, the classical oracle, the loaded amplitudes, and the comparator state."""
    from qiskit.circuit.library import IntegerComparator, StatePreparation
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation

    pmf, centers, lo, hi = discretise_losses(losses, n_qubits)
    var_c = float(np.quantile(losses, alpha))
    tailmask = centers >= var_c
    cvar_c = float(centers[tailmask & (pmf > 0)].dot(pmf[tailmask & (pmf > 0)]) /
                   max(pmf[tailmask].sum(), 1e-12))

    amps = np.sqrt(pmf); amps = amps / (np.linalg.norm(amps) or 1.0)
    loader = StatePreparation(amps)
    spb = (hi - lo) / (2 ** n_qubits - 1)
    ti = int(np.clip(round((var_c - lo) / spb), 1, 2 ** n_qubits - 1))
    comp = IntegerComparator(n_qubits, ti, geq=True)
    nq = comp.num_qubits
    qc = QuantumCircuit(nq)
    qc.append(loader, range(n_qubits))
    qc.append(comp, range(nq))
    problem = EstimationProblem(state_preparation=qc, objective_qubits=[n_qubits])
    iqae = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=0.05, sampler=StatevectorSampler())
    est = iqae.estimate(problem)
    p_tail = float(est.estimation)

    tail = np.arange(2 ** n_qubits) >= ti
    tp = pmf[tail].sum()
    cvar_q = float((pmf[tail] * centers[tail]).sum() / tp) if tp > 0 else var_c
    return {"var": var_c, "cvar_classical": cvar_c, "cvar_quantum": cvar_q,
            "p_tail_quantum": p_tail, "p_tail_classical": float(tp), "target_tail": 1 - alpha,
            "pmf": pmf, "centers": centers, "amplitudes": amps, "threshold_index": ti,
            "rel_err_cvar": abs(cvar_q - cvar_c) / (abs(cvar_c) + 1e-9),
            "rel_err_tail": abs(p_tail - tp) / (tp + 1e-9), "n_qubits": n_qubits, "alpha": alpha}
