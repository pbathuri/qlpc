"""Generate fixed-grid reference distributions; no mitigation outcome selection.

Run: python generate_distributions.py [--refresh-device]
The default reuses the frozen device model; refresh explicitly re-extracts the
installed FakeKingston snapshot and should be treated as a changed experiment.
"""
import argparse
import hashlib
import json
import time

import numpy as np
from qiskit import QuantumCircuit

from device_circuits import (ROOT, build_tfim, compile_native, dump_json,
    exact_probabilities, fold_cz, freeze_device, ideal_probabilities, load_device,
    noise_from_serialized, noisy_probabilities, readout_matrices, simulator)


def run(refresh=False):
    started = time.perf_counter()
    if refresh or not (ROOT/'data/device_noise.json').exists():
        model, calibration, serialized = freeze_device()
    else:
        model, calibration, serialized = load_device()
    engine = simulator(model)
    without_cz = {'errors': [e for e in serialized['errors']
                   if not (e['type'] == 'qerror' and 'cz' in e['operations'])]}
    diagnostic_engine = simulator(noise_from_serialized(without_cz))
    arrays = {}
    records = {'device': calibration, 'cases': [], 'calibrations': {},
               'basis_order': 'integer basis labels, qubit 0 least significant',
               'cz_noise_off_role': 'diagnostic-only CZ-error-disabled endpoint, not an extrapolation fit point'}
    checks = {'benchmark': {}, 'cases': [], 'aer_forward_sample': {}}
    # Benchmark worst-case before the complete sweep.
    bench = fold_cz(compile_native(build_tfim(8,1.5,8)),5)
    t = time.perf_counter()
    noisy_probabilities(bench, engine, readout_matrices(serialized,8))
    checks['benchmark'] = {'n_qubits':8, 'steps':8, 'time':1.5,
                           'fold':5, 'seconds':time.perf_counter()-t}
    print('Benchmark:',checks['benchmark'],flush=True)
    index = 0
    for n in (4,6,8):
        matrices = readout_matrices(serialized,n)
        for label in ('zero','one'):
            circuit = QuantumCircuit(n)
            if label == 'one':
                circuit.x(range(n))
            arrays[f'cal{n}_{label}'], _ = noisy_probabilities(circuit, engine, matrices)
        records['calibrations'][str(n)] = {
            'zero_key':f'cal{n}_zero','one_key':f'cal{n}_one',
            'state_preparation':'zero initial state or one noisy X per qubit',
            'readout_assignment_matrices_observed_rows_prepared_columns':[m.tolist() for m in matrices]}
        for duration in (0.5,1.0,1.5):
            continuous = exact_probabilities(n,duration)
            for steps in (2,4,8):
                native = compile_native(build_tfim(n,duration,steps))
                ideal = ideal_probabilities(native)
                # Ensure optimization/routing did not change logical order/state.
                uncompiled = ideal_probabilities(build_tfim(n,duration,steps))
                compiler_error = float(np.max(np.abs(ideal-uncompiled)))
                if compiler_error > 1e-12:
                    raise ValueError('Transpilation changed logical probabilities')
                arrays[f'case{index}_ideal'] = ideal
                arrays[f'case{index}_continuous'] = continuous
                arrays[f'case{index}_cz_noise_off'], _ = noisy_probabilities(native,diagnostic_engine,matrices)
                record = {'index':index,'n_qubits':n,'time':duration,'steps':steps,
                          'base_cz':int(native.count_ops().get('cz',0)), 'folds':{},
                          'ideal_fold_max_error':0.0,'compiler_probability_max_error':compiler_error}
                noncz = [(x.operation.name, tuple(x.operation.params), tuple(native.find_bit(q).index for q in x.qubits)) for x in native.data if x.operation.name!='cz']
                for factor in (1,3,5):
                    folded = fold_cz(native,factor)
                    folded_noncz = [(x.operation.name,tuple(x.operation.params),tuple(folded.find_bit(q).index for q in x.qubits)) for x in folded.data if x.operation.name not in ('cz','barrier')]
                    if noncz != folded_noncz:
                        raise ValueError('Folding changed non-CZ sequence')
                    fold_error = float(np.max(np.abs(ideal_probabilities(folded)-ideal)))
                    if fold_error > 1e-12:
                        raise ValueError('Fold changed ideal state')
                    count = int(folded.count_ops().get('cz',0))
                    if count != factor * record['base_cz']:
                        raise ValueError('Wrong CZ amplification')
                    record['ideal_fold_max_error'] = max(record['ideal_fold_max_error'],fold_error)
                    record['folds'][str(factor)] = {'cz':count,'depth':folded.depth(),
                        'one_qubit_count':sum(v for k,v in folded.count_ops().items() if k not in ('cz','barrier'))}
                    noisy, pre = noisy_probabilities(folded,engine,matrices)
                    arrays[f'case{index}_fold{factor}'] = noisy
                    arrays[f'case{index}_fold{factor}_prereadout'] = pre
                    if index == 0 and factor == 3:
                        measured = folded.copy()
                        measured.measure_all()
                        shots = 131072
                        counts = engine.run(measured,shots=shots,seed_simulator=24680).result().get_counts()
                        sampled = np.zeros(2**n)
                        for bitstring, count in counts.items():
                            sampled[int(bitstring,2)] = count/shots
                        # Each bin is checked using a conservative seven-sigma
                        # absolute tolerance plus rare-bin finite-shot allowance.
                        tolerance = 7*np.sqrt(noisy*(1-noisy)/shots)+7/shots
                        residual = float(np.max(np.abs(sampled-noisy)/tolerance))
                        if residual > 1:
                            raise ValueError('Aer measurement/readout forwarding disagrees')
                        checks['aer_forward_sample'] = {'case':0,'fold':3,'shots':shots,
                            'seed':24680,'max_fraction_of_seven_sigma_tolerance':residual,
                            'max_absolute_probability_error':float(np.max(np.abs(sampled-noisy)))}
                records['cases'].append(record)
                checks['cases'].append({'index':index,'ideal_fold_max_error':record['ideal_fold_max_error'],
                    'compiler_probability_max_error':compiler_error,'noncz_sequence_unchanged':True,
                    'cz_scaling_exact':True,'compact_register':n})
                print(f'case {index:02d}: n={n}, T={duration}, steps={steps}',flush=True)
                index += 1
    np.savez_compressed(ROOT/'data/distributions.npz',**arrays)
    records['distribution_sha256'] = hashlib.sha256((ROOT/'data/distributions.npz').read_bytes()).hexdigest()
    checks['elapsed_seconds'] = time.perf_counter()-started
    checks['case_count'] = len(records['cases'])
    checks['noise_sha256'] = calibration['noise_sha256']
    dump_json(ROOT/'data/circuits.json',records)
    dump_json(ROOT/'verification/device_checks.json',checks)
    print(f'Completed {index} cases in {checks["elapsed_seconds"]:.2f}s',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-device',action='store_true')
    run(parser.parse_args().refresh_device)
