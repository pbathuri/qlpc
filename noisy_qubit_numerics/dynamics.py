"""One-qubit Lindblad dynamics; time is in microseconds, angular rates in rad/us.

The matrix-exponential reference is exact for this finite model up to roundoff.
Euler and RK4 are deliberately not projected onto physical density matrices.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)


@dataclass(frozen=True)
class Model:
    """Driven qubit with zero-temperature relaxation and Markovian dephasing."""

    omega: float = 0.6
    detuning: float = 0.2
    t1_us: float = 50.0
    t2_us: float = 40.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.omega, self.detuning)):
            raise ValueError("Angular frequencies must be finite.")
        if any(math.isnan(v) or v <= 0 for v in (self.t1_us, self.t2_us)):
            raise ValueError("T1 and T2 must be positive (infinity is allowed).")
        if self.t2_us > 2 * self.t1_us:
            raise ValueError("T2 must not exceed 2*T1 for nonnegative pure dephasing.")

    @property
    def gamma1(self):
        return 1.0 / self.t1_us

    @property
    def gamma_phi(self):
        return 1.0 / self.t2_us - self.gamma1 / 2

    @property
    def hamiltonian(self):
        return (self.omega * X + self.detuning * Z) / 2


def _initial(initial):
    rho = np.diag([1.0, 0.0]).astype(complex) if initial is None else np.array(initial, dtype=complex)
    if rho.shape != (2, 2) or not np.all(np.isfinite(rho)):
        raise ValueError("Initial state must be a finite 2 by 2 density matrix.")
    if not np.allclose(rho, rho.conj().T, atol=1e-12, rtol=0):
        raise ValueError("Initial state must be Hermitian.")
    if abs(np.trace(rho) - 1) > 1e-12 or np.linalg.eigvalsh(rho).min() < -1e-12:
        raise ValueError("Initial state must have trace one and be positive semidefinite.")
    return rho


def _time(time_us):
    if not math.isfinite(time_us) or time_us < 0:
        raise ValueError("Time must be finite and nonnegative.")


def liouvillian(model):
    """Return the 4 by 4 generator acting on column-stacked density matrices."""
    h = model.hamiltonian
    generator = -1j * (np.kron(I2, h) - np.kron(h.T, I2))
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    for collapse in (np.sqrt(model.gamma1) * lowering, np.sqrt(model.gamma_phi / 2) * Z):
        product = collapse.conj().T @ collapse
        generator += np.kron(collapse.conj(), collapse)
        generator -= (np.kron(I2, product) + np.kron(product.T, I2)) / 2
    return generator


def reference_expm(model, time_us, initial=None):
    """Solve the autonomous Lindblad system by its matrix exponential."""
    _time(time_us)
    state = expm(time_us * liouvillian(model)) @ _initial(initial).reshape(4, order="F")
    return state.reshape((2, 2), order="F")


def reference_bloch(model, time_us, initial=None):
    """Independent affine Bloch-coordinate reference; no Liouvillian reuse."""
    _time(time_us)
    rho = _initial(initial)
    x, y, z = [np.trace(rho @ pauli).real for pauli in (X, Y, Z)]
    generator = np.array([
        [-1 / model.t2_us, -model.detuning, 0, 0],
        [model.detuning, -1 / model.t2_us, -model.omega, 0],
        [0, model.omega, -1 / model.t1_us, 1 / model.t1_us],
        [0, 0, 0, 0],
    ])
    x, y, z, _ = expm(time_us * generator) @ np.array([x, y, z, 1])
    return (I2 + x * X + y * Y + z * Z) / 2


def integrate(model, duration_us, steps, method, initial=None):
    """Return all states, including the initial state, without positivity repair.

    Methods: forward Euler, classical RK4, or Strang splitting. The latter
    composes exact half-unitary steps with an exact amplitude/dephasing channel.
    The dissipative subchannels commute for this specific one-qubit model.
    """
    _time(duration_us)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if method not in ("euler", "rk4", "strang"):
        raise ValueError("method must be euler, rk4, or strang.")
    dt = duration_us / steps
    trajectory = np.empty((steps + 1, 2, 2), dtype=complex)
    trajectory[0] = _initial(initial)
    if method == "strang":
        half_unitary = expm(-1j * model.hamiltonian * dt / 2)
        survival = np.exp(-model.gamma1 * dt)
        coherence = np.exp(-(model.gamma1 / 2 + model.gamma_phi) * dt)
        for k in range(steps):
            rho = half_unitary @ trajectory[k] @ half_unitary.conj().T
            damped = np.array([
                [rho[0, 0] + (1 - survival) * rho[1, 1], coherence * rho[0, 1]],
                [coherence * rho[1, 0], survival * rho[1, 1]],
            ])
            trajectory[k + 1] = half_unitary @ damped @ half_unitary.conj().T
    else:
        generator = liouvillian(model)
        for k in range(steps):
            state = trajectory[k].reshape(4, order="F")
            k1 = generator @ state
            if method == "euler":
                state = state + dt * k1
            else:
                k2 = generator @ (state + dt * k1 / 2)
                k3 = generator @ (state + dt * k2 / 2)
                k4 = generator @ (state + dt * k3)
                state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            trajectory[k + 1] = state.reshape((2, 2), order="F")
    return trajectory


def trace_error(rho, reference):
    """Half nuclear norm of the error, also defined for nonphysical iterates."""
    return float(np.linalg.svd(np.asarray(rho) - np.asarray(reference), compute_uv=False).sum() / 2)


def p1(rho):
    """Excited-state population; nonphysical iterates are not clipped."""
    return float(np.asarray(rho)[1, 1].real)
