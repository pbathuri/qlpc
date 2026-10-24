"""Compact, calibration-derived TFIM circuits and exact noisy distributions.

All vectors use integer basis labels: bit j represents local qubit j.
Calibration is a dated device snapshot; this is not a hardware execution.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError
from scipy.linalg import eigh

BASIS = ['rz', 'sx', 'x', 'cz', 'id']
ROOT = Path(__file__).resolve().parent


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')


def noise_from_serialized(data):
    """Rehydrate JSON complex-pair matrices via public channel constructors."""
    model = NoiseModel(basis_gates=BASIS)
    for source in data['errors']:
        error = copy.deepcopy(source)
        if error['type'] == 'qerror':
            for sequence in error['instructions']:
                for instruction in sequence:
                    if instruction['name'] in ('kraus', 'unitary'):
                        array = np.asarray(instruction['params'])
                        if array.ndim == 4 and array.shape[-1] == 2:
                            instruction['params'] = list(array[..., 0] + 1j * array[..., 1])
            channel = QuantumError.from_dict(error)
            # Aer drops tiny branch probabilities in decomposed error circuits.
            # A dense SuperOp retains the same channel without that truncation.
            channel = QuantumError([(channel.to_quantumchannel().to_instruction(), 1.0)])
            for qubits in error['gate_qubits']:
                model.add_quantum_error(channel, error['operations'], qubits)
        elif error['type'] == 'roerror':
            for qubits in error['gate_qubits']:
                model.add_readout_error(ReadoutError(error['probabilities']), qubits)
        else:
            raise ValueError(error['type'])
    return model


def choose_chain(backend, length=8):
    """Lexicographically first simple path with usable native calibrations.

This deterministic topology-only policy is independent of experiment outcomes.
"""
    props = backend.properties()
    edges = set(map(tuple, backend.coupling_map.get_edges()))
    valid = []
    for q in range(backend.num_qubits):
        try:
            values = [props.t1(q), props.t2(q), props.readout_error(q)]
            values += [props.gate_error(g, [q]) for g in ('x', 'sx', 'id')]
            if props.is_qubit_operational(q) and all(np.isfinite(values)) and min(values[:2]) > 0 and max(values[2:]) < 0.5:
                valid.append(q)
        except Exception:
            continue
    valid = set(valid)
    adjacency = {q: [] for q in valid}
    for a, b in sorted(edges):
        if a in valid and b in valid:
            try:
                if props.is_gate_operational('cz', [a,b]) and np.isfinite(props.gate_error('cz', [a,b])) and props.gate_error('cz', [a,b]) < 0.5:
                    adjacency[a].append(b)
            except Exception:
                continue
    def visit(path):
        if len(path) == length:
            return path
        for nxt in sorted(adjacency[path[-1]]):
            if nxt not in path:
                result = visit(path + [nxt])
                if result:
                    return result
        return None
    for start in sorted(valid):
        result = visit([start])
        if result:
            return result
    raise ValueError('No usable device path found')


def freeze_device(data_dir=ROOT / 'data'):
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    backend = FakeKingston()
    chain = choose_chain(backend)
    mapping = {p: q for q, p in enumerate(chain)}
    source = NoiseModel.from_backend(backend).to_dict(serializable=True)
    remapped = {'errors': []}
    for error in source['errors']:
        groups = [g for g in error.get('gate_qubits', []) if all(p in mapping for p in g)]
        if not groups:
            continue
        # Only native gates used in this experiment plus measurement readout.
        if error['type'] == 'qerror' and not set(error['operations']).intersection(BASIS):
            continue
        local = copy.deepcopy(error)
        if local['type'] == 'roerror':
            # BackendV2 exposes only average readout_error to from_backend.
            # Preserve the richer directional data in the calibration file.
            if len(groups) != 1 or len(groups[0]) != 1:
                raise ValueError('Expected single-qubit assignment calibration')
            physical = groups[0][0]
            e0 = backend.properties().qubit_property(physical, 'prob_meas1_prep0')[0]
            e1 = backend.properties().qubit_property(physical, 'prob_meas0_prep1')[0]
            local['probabilities'] = [[1-e0, e0], [e1, 1-e1]]
        local['gate_qubits'] = [[mapping[p] for p in group] for group in groups]
        if 'id' in local:
            local.pop('id')
            local['id'] = hashlib.sha256(json.dumps(local, sort_keys=True).encode()).hexdigest()[:32]
        remapped['errors'].append(local)
    props = backend.properties().to_dict()
    calibration = {
        'backend': backend.name, 'calibration_date': str(props['last_update_date']),
        'physical_qubits': chain, 'selection_policy': 'lexicographically first operational calibrated eight-qubit CZ path; errors < 0.5',
        'logical_order': 'local qubit j maps to physical_qubits[j]; integer basis little-endian',
        'qubits': [{'physical': q, 'properties': props['qubits'][q]} for q in chain],
        'gates': [g for g in props['gates'] if all(q in mapping for q in g['qubits']) and g['gate'] in BASIS],
        'versions': {p: importlib.metadata.version(p) for p in ['qiskit','qiskit-aer','qiskit-ibm-runtime','numpy','scipy']},
        'source': 'qiskit_ibm_runtime.fake_provider.FakeKingston bundled properties; NoiseModel.from_backend gate channels with directional readout probabilities from BackendProperties',
        'readout_override': 'BackendV2 from_backend averages directional errors; independent ReadoutError channels here explicitly retain prob_meas1_prep0 and prob_meas0_prep1',
        'noise_scope': 'gate depolarization and thermal relaxation, independent assignment errors; no idle schedule, crosstalk, drift, leakage or hardware execution',
        'channel_representation': 'Dense SuperOp conversion consolidates Kraus branches into a probability-one channel and prevents Aer truncating rare branch probabilities',
        'source_files_sha256': {name: hashlib.sha256((Path(backend.dirname) / name).read_bytes()).hexdigest() for name in (backend.props_filename, backend.conf_filename)},
        'readout_rates': [{'physical': q, 'p_meas1_prep0': backend.properties().qubit_property(q, 'prob_meas1_prep0')[0], 'p_meas0_prep1': backend.properties().qubit_property(q, 'prob_meas0_prep1')[0]} for q in chain],
        'folding_scope': 'native CZ only; one-qubit errors remain fixed; fold-zero intercept does not remove all noise',
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    dump_json(data_dir / 'device_noise.json', remapped)
    calibration['noise_sha256'] = hashlib.sha256((data_dir / 'device_noise.json').read_bytes()).hexdigest()
    dump_json(data_dir / 'device_calibration.json', calibration)
    return noise_from_serialized(remapped), calibration, remapped


def load_device(data_dir=ROOT / 'data'):
    data = json.loads((data_dir / 'device_noise.json').read_text())
    calibration = json.loads((data_dir / 'device_calibration.json').read_text())
    if hashlib.sha256((data_dir / 'device_noise.json').read_bytes()).hexdigest() != calibration['noise_sha256']:
        raise ValueError('Frozen noise model checksum mismatch')
    return noise_from_serialized(data), calibration, data


def build_tfim(n, time, steps):
    circuit = QuantumCircuit(n)
    dt = time / steps
    # exp(-i dt X/2) exp(-i dt ZZ) exp(-i dt X/2).
    for _ in range(steps):
        for q in range(n):
            circuit.rx(0.7 * dt, q)
        for q in range(n - 1):
            circuit.rzz(2 * dt, q, q + 1)
        for q in range(n):
            circuit.rx(0.7 * dt, q)
    return circuit


def compile_native(circuit):
    n = circuit.num_qubits
    edges = [(i,j) for i in range(n) for j in range(n) if abs(i-j) == 1]
    return transpile(circuit, basis_gates=BASIS, coupling_map=CouplingMap(edges),
                     initial_layout=list(range(n)), optimization_level=1, seed_transpiler=0)


def fold_cz(circuit, factor):
    if factor not in (1, 3, 5):
        raise ValueError('Fold factor must be 1, 3 or 5')
    out = QuantumCircuit(circuit.num_qubits)
    out.global_phase = circuit.global_phase
    for item in circuit.data:
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        if item.operation.name == 'cz':
            for _ in range(factor):
                out.append(item.operation, qubits)
                out.barrier(*qubits)
        else:
            out.append(item.operation, qubits)
    return out


def exact_probabilities(n, time):
    dim = 2 ** n
    ids = np.arange(dim)
    z = 1 - 2 * ((ids[:,None] >> np.arange(n)) & 1)
    h = np.diag(np.sum(z[:,:-1] * z[:,1:], axis=1).astype(float))
    for q in range(n):
        h[ids, ids ^ (1 << q)] += 0.7
    values, vectors = eigh(h)
    state = vectors @ (np.exp(-1j * time * values) * vectors[0,:])
    return np.abs(state) ** 2


def readout_matrices(data, n):
    mats = [None] * n
    for error in data['errors']:
        if error['type'] == 'roerror':
            for group in error['gate_qubits']:
                if len(group) != 1:
                    raise ValueError('Expected independent readout errors')
                if group[0] < n:
                    # Aer stores P(measured | prepared) with prepared as row.
                    mats[group[0]] = np.asarray(error['probabilities']).T
    if any(m is None for m in mats):
        raise ValueError('Missing readout calibration')
    return mats


def apply_readout(probabilities, matrices):
    operator = np.array([[1.0]])
    for matrix in reversed(matrices):
        operator = np.kron(operator, matrix)
    return operator @ probabilities


def simulator(model):
    return AerSimulator(method='density_matrix', noise_model=model,
                        max_parallel_threads=2, max_parallel_experiments=1)


def noisy_probabilities(circuit, engine, matrices):
    measured = circuit.copy()
    measured.save_density_matrix()
    result = engine.run(measured, shots=1, seed_simulator=1729).result()
    if not result.success:
        raise RuntimeError(result.status)
    rho = np.asarray(result.data(0)['density_matrix'])
    diagonal = np.real(np.diag(rho))
    if diagonal.min() < -1e-12 or abs(diagonal.sum() - 1) > 1e-10:
        raise ValueError('Density matrix probability normalization failed')
    diagonal = np.maximum(diagonal, 0)
    diagonal /= diagonal.sum()
    return apply_readout(diagonal, matrices), diagonal


def ideal_probabilities(circuit):
    return Statevector.from_instruction(circuit).probabilities()
