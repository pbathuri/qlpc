"""Analytic, independent-reference, order, and physicality checks."""

import unittest
import numpy as np

from dynamics import Model, integrate, p1, reference_bloch, reference_expm, trace_error


class DynamicsTests(unittest.TestCase):
    def test_noiseless_rotation(self):
        model = Model(omega=0.6, detuning=0, t1_us=np.inf, t2_us=np.inf)
        for time in (0, 0.1, 2, 10):
            self.assertAlmostEqual(p1(reference_expm(model, time)), np.sin(0.6 * time / 2) ** 2, places=13)

    def test_analytic_relaxation_and_dephasing(self):
        model = Model(omega=0, detuning=0)
        excited = np.diag([0, 1])
        plus = np.ones((2, 2)) / 2
        for time in (0.1, 5, 50):
            self.assertAlmostEqual(p1(reference_expm(model, time, excited)), np.exp(-time / 50), places=13)
            rho = reference_expm(model, time, plus)
            self.assertAlmostEqual(rho[0, 1].real, 0.5 * np.exp(-time / 40), places=13)
            np.testing.assert_allclose(integrate(model, time, 1, "strang", plus)[-1], rho, atol=2e-14)

    def test_bloch_reference_agreement(self):
        rng = np.random.default_rng(702)
        for model in (Model(), Model(0.3, -0.7, np.inf, np.inf), Model(0, 0.5, 10, 20)):
            for time in (0, 0.7, 5, 40):
                v = rng.normal(size=2) + 1j * rng.normal(size=2)
                v /= np.linalg.norm(v)
                initial = np.outer(v, v.conj())
                # Different real/complex expm paths accumulate different roundoff
                # over the longest oscillatory case (40 us).
                np.testing.assert_allclose(reference_expm(model, time, initial), reference_bloch(model, time, initial), atol=3e-12, rtol=0)

    def test_trace_and_hermiticity(self):
        for method in ("euler", "rk4", "strang"):
            states = integrate(Model(), 10, 100, method)
            np.testing.assert_allclose(np.trace(states, axis1=1, axis2=2), 1, atol=2e-14, rtol=0)
            np.testing.assert_allclose(states, states.conj().transpose(0, 2, 1), atol=2e-14, rtol=0)

    def test_convergence_orders(self):
        model = Model()
        reference = reference_expm(model, 10)
        for method, expected in (("euler", 1), ("rk4", 4), ("strang", 2)):
            errors = [trace_error(integrate(model, 10, n, method)[-1], reference) for n in (160, 320, 640)]
            orders = np.log2(np.array(errors[:-1]) / errors[1:])
            np.testing.assert_allclose(orders, expected, atol=0.12, rtol=0, err_msg=method)

    def test_euler_positivity_failure_is_retained(self):
        states = integrate(Model(), 10, 5, "euler")
        self.assertLess(np.linalg.eigvalsh(states[1]).min(), -0.1)

    def test_strang_physical_at_every_step(self):
        for steps in (1, 2, 5, 100):
            states = integrate(Model(), 10, steps, "strang")
            self.assertGreaterEqual(np.linalg.eigvalsh(states).min(), -1e-13)
            np.testing.assert_allclose(np.trace(states, axis1=1, axis2=2), 1, atol=2e-14, rtol=0)

    def test_invalid_parameters(self):
        for kwargs in ({"t1_us": 0}, {"t2_us": -1}, {"t2_us": 101}, {"omega": np.nan}, {"t1_us": np.nan}, {"t2_us": np.inf}):
            with self.assertRaises(ValueError):
                Model(**kwargs)
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                integrate(Model(), 10, steps, "euler")
        with self.assertRaises(ValueError):
            integrate(Model(), 10, 10, "unknown")
        for time in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                reference_expm(Model(), time)
        for rho in (np.eye(3), np.eye(2), np.diag([2, -1]), [[1, 1], [0, 0]]):
            with self.assertRaises(ValueError):
                reference_expm(Model(), 1, rho)

    def test_zero_duration_and_input_preserved(self):
        initial = np.array([[0.5, -0.5j], [0.5j, 0.5]])
        before = initial.copy()
        for method in ("euler", "rk4", "strang"):
            states = integrate(Model(), 0, 3, method, initial)
            np.testing.assert_allclose(states, np.repeat(initial[None], 4, axis=0), atol=1e-15)
        np.testing.assert_array_equal(initial, before)


if __name__ == "__main__":
    unittest.main()
