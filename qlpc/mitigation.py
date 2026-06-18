"""Quantum error MITIGATION — NISQ-relevant, and honest about the QEC boundary.

Full quantum error *correction* (surface codes, etc.) needs fault-tolerant hardware with ~10^3 physical
qubits per logical qubit — not available for these circuits today. At NISQ scale and on simulators, the
relevant, proven technique is error *mitigation*. We demonstrate **readout error mitigation with mthree**
(M3 — Nation, Kang, Sundaresan, Gambetta 2021): on a noisy `qiskit-aer` simulator it recovers a quantum
estimate that readout + depolarizing noise corrupts. (Fault-tolerant QEC is noted as the future path.)
"""
from __future__ import annotations

import numpy as np


def readout_mitigation_demo(theta: float = 1.2, shots: int = 8192, seed: int = 7) -> dict:
    """Prepare Ry(theta)|0> (known P(|1>) = sin^2(theta/2)); measure ideal vs noisy vs mthree-mitigated.
    Returns the three probabilities + the error reduction from mitigation."""
    p_true = float(np.sin(theta / 2) ** 2)
    out = {"target_p1": round(p_true, 4)}
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

        qc = QuantumCircuit(1, 1)
        qc.ry(theta, 0)
        qc.measure(0, 0)

        # NISQ noise: 2% gate depolarizing + asymmetric readout error
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 1), ["ry"])
        nm.add_all_qubit_readout_error(ReadoutError([[0.93, 0.07], [0.12, 0.88]]))
        backend = AerSimulator(noise_model=nm, seed_simulator=seed)

        tqc = transpile(qc, backend)
        counts = backend.run(tqc, shots=shots).result().get_counts()
        p_noisy = counts.get("1", 0) / shots

        import mthree
        mit = mthree.M3Mitigation(backend)
        mit.cals_from_system(qubits=[0], shots=shots)
        quasi = mit.apply_correction(counts, qubits=[0])
        p_mit = float(quasi.get("1", 0.0))

        out.update(method="mthree-readout-mitigation",
                   p_noisy=round(p_noisy, 4), p_mitigated=round(p_mit, 4),
                   err_noisy=round(abs(p_noisy - p_true), 4), err_mitigated=round(abs(p_mit - p_true), 4),
                   error_reduction=round((abs(p_noisy - p_true) - abs(p_mit - p_true)) /
                                         (abs(p_noisy - p_true) + 1e-9), 3),
                   note="NISQ mitigation; fault-tolerant QEC is the future hardware path")
        return out
    except Exception as e:  # noqa: BLE001
        out.update(method="unavailable", note=f"{type(e).__name__}: {str(e)[:80]}")
        return out
