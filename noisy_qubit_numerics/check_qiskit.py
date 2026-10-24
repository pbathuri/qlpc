"""Optional Qiskit/Aer cross-check: invocation fails if dependencies/checks fail.

Run with a Qiskit/Aer environment; JSON is printed to stdout. This reconstructs
the original QLPC one-qubit noise baseline without requiring M3. It checks
forward noise channels, not the original project's mitigation output.
"""

import json
import numpy as np
from scipy.linalg import expm
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, ReadoutError, amplitude_damping_error,
    depolarizing_error, phase_damping_error,
)

from dynamics import Model, integrate, trace_error


def main():
    differences = []
    for model in (Model(), Model(0.3, -0.5, 12, 20), Model(0.6, 0.2, np.inf, np.inf)):
        for steps in (1, 5, 40):
            dt = 10 / steps
            half_unitary = Operator(expm(-1j * model.hamiltonian * dt / 2))
            amplitude = amplitude_damping_error(1 - np.exp(-dt / model.t1_us)).to_quantumchannel()
            # Aer phase damping multiplies coherences by sqrt(1 - parameter).
            phase = phase_damping_error(1 - np.exp(-2 * model.gamma_phi * dt)).to_quantumchannel()
            state = DensityMatrix.from_label("0")
            trajectory = integrate(model, 10, steps, "strang")
            for index in range(steps):
                state = state.evolve(half_unitary).evolve(amplitude).evolve(phase).evolve(half_unitary)
                differences.append(trace_error(state.data, trajectory[index + 1]))
    channel_error = max(differences)
    if channel_error > 1e-12:
        raise AssertionError(f"Strang/Qiskit channel discrepancy: {channel_error}")

    angle, strength, alpha, beta = 1.2, 0.02, 0.07, 0.12
    ideal = QuantumCircuit(1)
    ideal.ry(angle, 0)
    ideal_state = DensityMatrix.from_instruction(ideal)
    channel_state = ideal_state.evolve(depolarizing_error(strength, 1).to_quantumchannel())
    expected_state = (1 - strength) * ideal_state.data + strength * np.eye(2) / 2
    np.testing.assert_allclose(channel_state.data, expected_state, atol=1e-14, rtol=0)
    p_ideal = float(np.sin(angle / 2) ** 2)
    p_noisy = (1 - strength) * p_ideal + strength / 2
    q_expected = alpha * (1 - p_noisy) + (1 - beta) * p_noisy
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(strength, 1), ["ry"])
    noise.add_all_qubit_readout_error(ReadoutError([[1 - alpha, alpha], [beta, 1 - beta]]))
    simulator = AerSimulator(method="density_matrix", noise_model=noise)
    saved = ideal.copy()
    saved.save_density_matrix()
    aer_state = simulator.run(transpile(saved, simulator, optimization_level=0)).result().data(0)["density_matrix"]
    np.testing.assert_allclose(np.asarray(aer_state), expected_state, atol=1e-14, rtol=0)
    measured = ideal.copy()
    measured.measure_all()
    shots, seed = 200000, 20261024
    counts = simulator.run(transpile(measured, simulator, optimization_level=0), shots=shots, seed_simulator=seed).result().get_counts()
    q_measured = counts.get("1", 0) / shots
    sigma = float(np.sqrt(q_expected * (1 - q_expected) / shots))
    z_score = (q_measured - q_expected) / sigma
    if abs(z_score) > 6:
        raise AssertionError(f"Readout sample differs from expectation by {z_score} standard deviations.")
    print(json.dumps({
        "status": "passed", "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "strang_channel_comparisons": len(differences),
        "strang_max_trace_error": channel_error,
        "qlpc_baseline": {"ry_angle_radians": angle, "depolarizing_parameter": strength,
                          "alpha": alpha, "beta": beta, "p_ideal": p_ideal,
                          "p_after_depolarizing": p_noisy, "readout_expected": q_expected,
                          "readout_sampled": q_measured, "shots": shots,
                          "seed_simulator": seed, "standard_errors_from_expected": z_score},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
