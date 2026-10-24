"""Independent device-channel, ordering and evolution checks (unittest)."""
import json
import unittest

import numpy as np
from numpy.testing import assert_allclose
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SuperOp
from qiskit_aer.noise import NoiseModel, QuantumError
from qiskit_ibm_runtime.fake_provider import FakeKingston
from scipy.linalg import expm

from device_circuits import (ROOT, apply_readout, build_tfim, choose_chain,
    compile_native, exact_probabilities, fold_cz, ideal_probabilities,
    load_device, noisy_probabilities, readout_matrices, simulator)


class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.calibration, cls.serialized = load_device()
        cls.backend = FakeKingston()
        cls.source = NoiseModel.from_backend(cls.backend).to_dict()

    def test_calibration_date_and_path(self):
        self.assertTrue(self.calibration['calibration_date'].startswith('2026-04-15'))
        self.assertEqual(self.calibration['physical_qubits'],choose_chain(self.backend))
        self.assertEqual(self.calibration['physical_qubits'],list(range(8)))
        edges = set(map(tuple,self.backend.coupling_map.get_edges()))
        self.assertTrue(all((a,b) in edges for a,b in zip(range(7),range(1,8))))

    def test_every_remapped_channel_matches_source(self):
        reference = {}
        for error in self.source['errors']:
            for group in error.get('gate_qubits',[]):
                reference[(error['type'],tuple(error['operations']),tuple(group))] = error
        tested = 0
        chain = self.calibration['physical_qubits']
        for error in self.model.to_dict()['errors']:
            for group in error['gate_qubits']:
                physical = tuple(chain[q] for q in group)
                src = reference[(error['type'],tuple(error['operations']),physical)]
                if error['type'] == 'qerror':
                    # The dense-channel constructor consolidates the Kraus
                    # branches into one probability-one channel instruction.
                    source = np.asarray(QuantumError.from_dict(src).to_quantumchannel())
                    current = np.asarray(QuantumError.from_dict(error).to_quantumchannel())
                    assert_allclose(current,source,atol=3e-14,rtol=0)
                    dimension = int(round(np.sqrt(len(source))))
                    identity = np.eye(dimension).reshape(-1)
                    assert_allclose(source.conj().T @ identity,identity,atol=3e-14,rtol=0)
                else:
                    e0=self.backend.properties().qubit_property(physical[0],'prob_meas1_prep0')[0]
                    e1=self.backend.properties().qubit_property(physical[0],'prob_meas0_prep1')[0]
                    assert_allclose(error['probabilities'],[[1-e0,e0],[e1,1-e1]],atol=0,rtol=0)
                tested += 1
        self.assertEqual(tested,len(self.serialized['errors']))
        self.assertGreater(tested,40)

    def test_compact_native_register_and_complete_noise(self):
        available = {(op,tuple(group)) for e in self.serialized['errors']
                     for op in e['operations'] for group in e['gate_qubits']}
        for n in (4,6,8):
            circuit = compile_native(build_tfim(n,1.,4))
            self.assertEqual(circuit.num_qubits,n)
            for item in circuit.data:
                if item.operation.name == 'rz':
                    continue  # Virtual rotations have no calibrated gate error.
                key = (item.operation.name,tuple(circuit.find_bit(q).index for q in item.qubits))
                self.assertIn(key,available)

    def test_readout_little_endian_orientation(self):
        matrices = [np.array([[.9,.2],[.1,.8]]),np.array([[.7,.4],[.3,.6]])]
        for integer in range(4):
            p = np.eye(4)[integer]
            expected = np.array([matrices[0][y&1,integer&1]*matrices[1][y>>1,integer>>1] for y in range(4)])
            assert_allclose(apply_readout(p,matrices),expected,atol=1e-15)

    def test_densitymatrix_unit_trace(self):
        circ = fold_cz(compile_native(build_tfim(8,1.5,8)),5)
        circ.save_density_matrix()
        result = simulator(self.model).run(circ,shots=1).result()
        rho = np.asarray(result.data(0)['density_matrix'])
        assert_allclose(np.trace(rho),1.,atol=2e-12,rtol=0)
        assert_allclose(rho,rho.conj().T,atol=2e-12,rtol=0)
        self.assertGreater(np.linalg.eigvalsh(rho).min(),-1e-12)

    def test_all_fixed_grid_ideal_folded_states(self):
        count = 0
        for n in (4,6,8):
            for time in (.5,1.,1.5):
                for steps in (2,4,8):
                    circ = compile_native(build_tfim(n,time,steps))
                    ideal = np.asarray(Statevector.from_instruction(circ))
                    for factor in (1,3,5):
                        folded = fold_cz(circ,factor)
                        assert_allclose(np.asarray(Statevector.from_instruction(folded)),ideal,atol=2e-13,rtol=0)
                        self.assertEqual(folded.count_ops()['cz'],factor*circ.count_ops()['cz'])
                    count += 1
        self.assertEqual(count,27)

    def test_exact_evolution_against_independent_kronecker(self):
        x=np.array([[0,1],[1,0]])
        z=np.diag([1,-1])
        identity=np.eye(2)
        h=np.kron(z,z)+.7*(np.kron(x,identity)+np.kron(identity,x))
        initial=np.array([1.,0.,0.,0.])
        for duration in (.5,1.,1.5):
            expected=np.abs(expm(-1j*duration*h)@initial)**2
            assert_allclose(exact_probabilities(2,duration),expected,atol=1e-14,rtol=0)
            assert_allclose(exact_probabilities(1,duration),[np.cos(.7*duration)**2,np.sin(.7*duration)**2],atol=1e-14)

    def test_strang_convergence_in_asymptotic_region(self):
        exact=exact_probabilities(4,.5)
        errors=[np.linalg.norm(ideal_probabilities(build_tfim(4,.5,s))-exact) for s in (8,16,32)]
        self.assertGreater(errors[0]/errors[1],3.9)
        self.assertGreater(errors[1]/errors[2],3.9)

    def test_saved_grid_and_calibration_probability_vectors(self):
        records=json.loads((ROOT/'data/circuits.json').read_text())
        checks=json.loads((ROOT/'verification/device_checks.json').read_text())
        self.assertEqual(len(records['cases']),27)
        self.assertLess(checks['aer_forward_sample']['max_fraction_of_seven_sigma_tolerance'],1)
        with np.load(ROOT/'data/distributions.npz') as data:
            for key in data.files:
                self.assertGreaterEqual(data[key].min(),0)
                assert_allclose(data[key].sum(),1,atol=1e-12)
            for n in (4,6,8):
                matrices=readout_matrices(self.serialized,n)
                expected=apply_readout(np.eye(2**n)[0],matrices)
                assert_allclose(data[f'cal{n}_zero'],expected,atol=1e-14)
                # Noisy X preparation deliberately prevents treating all-one
                # calibration as pure readout, and is retained rather than idealized.
                pure=apply_readout(np.eye(2**n)[-1],matrices)
                self.assertGreater(np.max(abs(data[f'cal{n}_one']-pure)),1e-5)


if __name__=='__main__':
    unittest.main()
