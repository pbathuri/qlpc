"""Device-agnostic fast QAOA statevector simulator.

For a portfolio QUBO the cost Hamiltonian H_C is DIAGONAL, so QAOA reduces to:
  * cost layer  : elementwise phase  psi *= exp(-i gamma * diag)
  * X-mixer     : per-qubit RX(2 beta) applied by pairing amplitudes that differ in one bit
No circuit synthesis per evaluation -> orders of magnitude faster than gate-based simulation, which is
what makes a many-instance, statistically-honest study tractable.

Backend selection:
  * CPU  -> numpy            (default; tractable to ~22-24 qubits on a workstation)
  * GPU  -> cupy             (RTX 3090 ~30 qubits; A100 80GB on BigRed200 ~33 qubits)
Set QLPC_DEVICE=gpu to request cupy; falls back to numpy with a recorded note if unavailable.
"""
from __future__ import annotations

import os

import numpy as np


def get_xp(prefer_gpu: bool | None = None):
    """Return (array_module, device_str). cupy if GPU requested and importable, else numpy."""
    want = (prefer_gpu if prefer_gpu is not None else os.environ.get("QLPC_DEVICE", "cpu").lower() == "gpu")
    if want:
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            return cp, "gpu"
        except Exception:  # noqa: BLE001
            return np, "cpu(gpu-unavailable)"
    return np, "cpu"


def cost_diagonal(mu, sigma, budget, q, penalty=None, xp=np, dtype=np.float64):
    """The 2^n diagonal of C(x) = -mu·x + q xᵀΣx + P(Σx - B)², vectorized over all basis states."""
    mu = np.asarray(mu, float); sigma = np.asarray(sigma, float); n = len(mu)
    if penalty is None:
        penalty = float(np.abs(mu).max() + q * np.abs(sigma).sum(axis=1).max() + 0.25)
    # bit matrix B[x, i] = bit i of x, for x in 0..2^n-1  (i is qubit/asset index, little-endian)
    xs = np.arange(2 ** n, dtype=np.int64)
    bits = ((xs[:, None] >> np.arange(n)[None, :]) & 1).astype(dtype)        # (2^n, n)
    lin = -(bits @ mu) + q * np.einsum("xi,ij,xj->x", bits, sigma, bits, optimize=True)
    card = bits.sum(axis=1)
    diag = lin + penalty * (card - budget) ** 2
    return xp.asarray(diag.astype(dtype)), penalty


def qaoa_probs(diag, gammas, betas, xp=np):
    """Probabilities |psi|² after the QAOA ansatz (uniform start, p layers). diag is an xp array (2^n)."""
    dim = diag.shape[0]
    n = int(round(np.log2(dim)))
    psi = xp.full(dim, 1.0 / np.sqrt(dim), dtype=xp.complex64)
    for gamma, beta in zip(gammas, betas):
        psi = psi * xp.exp(-1j * xp.asarray(np.float32(gamma)) * diag.astype(xp.float32))
        cb, sb = np.float32(np.cos(beta)), np.float32(np.sin(beta))
        for qb in range(n):                                  # RX(2β) on qubit qb: mix bit-paired amps
            psi = psi.reshape(-1, 2, 2 ** qb)
            a0, a1 = psi[:, 0, :], psi[:, 1, :]              # views; compute both new blocks before assign
            new0 = cb * a0 - 1j * sb * a1
            new1 = cb * a1 - 1j * sb * a0
            psi[:, 0, :] = new0
            psi[:, 1, :] = new1
            psi = psi.reshape(dim)
    p = xp.abs(psi) ** 2
    return p / p.sum()


def expectation(diag, probs, xp=np):
    return float((probs * diag).sum())


def to_numpy(a, xp=np):
    return a.get() if xp.__name__ == "cupy" else np.asarray(a)


def optimize_qaoa(diag, reps, restarts=10, maxiter=150, seed=0, xp=np):
    """Restarted COBYLA over (gammas, betas) minimizing ⟨H_C⟩. Returns (best_gammas, best_betas, history)."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    best_e, best_x, best_hist = np.inf, None, []
    for _ in range(max(1, restarts)):
        hist = []

        def energy(theta, _h=hist):
            p = qaoa_probs(diag, theta[:reps], theta[reps:], xp=xp)
            e = expectation(diag, p, xp=xp); _h.append(e); return e

        x0 = rng.uniform(0, 2 * np.pi, size=2 * reps)
        res = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter})
        if res.fun < best_e:
            best_e, best_x, best_hist = res.fun, res.x, hist
    return best_x[:reps], best_x[reps:], best_hist
