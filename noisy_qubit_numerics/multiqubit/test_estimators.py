import unittest
import numpy as np
from estimators import (FACTORS, LINEAR, QUADRATIC, METHODS, bit_table,
                        observable_weights, corrected_weights, calibration_rates,
                        shot_plan, nested_counts, estimate_counts, extrapolate)


class EstimatorTests(unittest.TestCase):
    def test_little_endian_and_observables(self):
        bits = bit_table(4)
        np.testing.assert_array_equal(bits[1], [1,0,0,0])
        w = observable_weights(4)
        np.testing.assert_array_equal(w[0], [1,1])
        np.testing.assert_array_equal(w[-1], [-1,1])
        self.assertEqual(w[1,0], .5)
        self.assertAlmostEqual(w[1,1], 1/3)

    def test_product_readout_inverse_correlated_state(self):
        n=4; bits=bit_table(n); p=np.zeros(2**n);p[[0,-1]]=[.4,.6]
        e0=np.array([.03,.06,.07,.02]);e1=np.array([.1,.04,.08,.09])
        observed=np.zeros_like(p)
        for initial in range(2**n):
            for output in range(2**n):
                chance=1.
                for q in range(n):
                    chance *= ((1-e0[q]) if bits[output,q]==0 else e0[q]) if bits[initial,q]==0 else (e1[q] if bits[output,q]==0 else (1-e1[q]))
                observed[output] += p[initial]*chance
        np.testing.assert_allclose(observed@corrected_weights(e0,e1)[0],p@observable_weights(n),atol=1e-14)

    def test_no_clipping(self):
        w=corrected_weights(np.array([.2,.2]),np.array([.2,.2]))
        self.assertGreater(w[0,0,0],1)
        self.assertGreater(w[0,0,1],1)

    def test_singular_rows_flagged(self):
        w=corrected_weights([[.5,.1],[.1,.1]],[[.5,.1],[.1,.1]])
        self.assertTrue(np.isnan(w[0]).all())
        self.assertTrue(np.isfinite(w[1]).all())

    def test_calibration_preparation_is_observed(self):
        zero=np.array([[80,10,5,5]])
        one=np.array([[5,5,10,80]])
        e0,e1,g=calibration_rates(zero,one,100,2)
        np.testing.assert_allclose(e0,[[.15,.10]])
        np.testing.assert_allclose(e1,[[.15,.10]])
        np.testing.assert_array_equal(g,[[70,80]])

    def test_budget_all_protocols_equal(self):
        plans=shot_plan()
        for name in METHODS:
            self.assertEqual(sum(plans[name])+(4096 if name.startswith('readout') else 0),32768)
        self.assertEqual(plans['zne_linear'],[10923,10923,10922])
        self.assertEqual(plans['readout_zne_linear'],[9558,9557,9557])
        with self.assertRaises(ValueError):shot_plan(4096)

    def test_nested_samples_keep_marginal_counts(self):
        counts=nested_counts(np.random.default_rng(42),[.3,.7],[10,30,20,10],100)
        for n,c in counts.items():np.testing.assert_array_equal(c.sum(axis=1),n)
        self.assertTrue(np.all(counts[30]>=counts[20]))
        self.assertTrue(np.all(counts[20]>=counts[10]))

    def test_polynomial_intercepts_and_coefficients(self):
        x=np.array(FACTORS)
        np.testing.assert_allclose(LINEAR@np.column_stack([np.ones(3),x]),[1,0],atol=1e-14)
        np.testing.assert_allclose(QUADRATIC@np.column_stack([np.ones(3),x,x*x]),[1,0,0],atol=1e-14)
        self.assertAlmostEqual(extrapolate(.4+.03*x,1),.4)
        self.assertAlmostEqual(extrapolate(.4+.03*x+.01*x*x,2),.4)
        self.assertGreater(np.linalg.norm(QUADRATIC),np.linalg.norm(LINEAR))

    def test_estimate_counts_matches_direct_sum(self):
        c=np.array([[3,2,4,1],[0,4,1,5]])
        w=observable_weights(2)
        np.testing.assert_allclose(estimate_counts(c,w),[sum(c[i,j]*w[j] for j in range(4))/10 for i in range(2)])
        np.testing.assert_allclose(estimate_counts(c,np.repeat(w[None],2,axis=0)),estimate_counts(c,w))


if __name__=='__main__':unittest.main()
