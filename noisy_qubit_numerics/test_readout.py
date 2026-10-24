"""Tests target algebra, confidence-set edge cases and correction failure modes."""
import unittest

import numpy as np
from scipy.stats import binom

from readout import (confidence_intervals, estimate_from_counts, hoeffding_radius,
                     inverse, invert_box, known_rate_interval, observed_probability)


class ReadoutTests(unittest.TestCase):
    def test_exact_inverse_targets_noisy_population(self):
        p_ideal = np.sin(1.2/2)**2
        p_noisy = .98*p_ideal + .01
        recovered = inverse(observed_probability(p_noisy, .07, .12), .07, .12)
        self.assertAlmostEqual(recovered, p_noisy)
        self.assertGreater(abs(recovered-p_ideal), .003)

    def test_unphysical_estimates_retained(self):
        self.assertLess(float(inverse(.01, .07, .12)), 0)
        self.assertGreater(float(inverse(.99, .07, .12)), 1)

    def test_singular_and_negative_estimated_gap_reported(self):
        estimates, gaps = estimate_from_counts([50, 50], 100, [50, 60], [50, 60], 100)
        self.assertTrue(np.isnan(estimates[0]))
        self.assertEqual(gaps[0], 0)
        self.assertLess(gaps[1], 0)
        self.assertAlmostEqual(estimates[1], .5)
        with self.assertRaises(ValueError):
            inverse(.5, .5, .5)

    def test_positive_gap_not_certified_uninformative(self):
        result = confidence_intervals(4096, 8192, 4014, 4014, 8192)
        self.assertTrue(result['positive_gap_not_certified'])
        self.assertEqual(result['lower'], 0)
        self.assertEqual(result['upper'], 1)

    def test_empty_physical_intersection_not_hidden(self):
        result = invert_box(.0, .01, .3, .4, .1, .2)
        self.assertTrue(result['empty'])
        self.assertGreater(result['lower'], result['upper'])

    def test_all_corners_enclose_interior_even_unphysical_corners(self):
        limits = [(.02, .6), (.05, .2), (.1, .3)]
        result = invert_box(*[v for pair in limits for v in pair])
        for q in np.linspace(*limits[0], 19):
            for e0 in np.linspace(*limits[1], 11):
                for e1 in np.linspace(*limits[2], 11):
                    p = inverse(q, e0, e1)
                    if 0 <= p <= 1:
                        self.assertLessEqual(result['lower'], p + 1e-14)
                        self.assertGreaterEqual(result['upper'], p - 1e-14)

    def test_known_rate_coverage_by_exact_binomial_enumeration(self):
        n = 80
        k = np.arange(n+1)
        intervals = known_rate_interval(k, n, .07, .12)
        for p in [0, .01, .1, .5, .9, .99, 1]:
            q = observed_probability(p, .07, .12)
            covered = (intervals['lower'] <= p) & (p <= intervals['upper'])
            self.assertGreaterEqual(binom.pmf(k, n, q)[covered].sum(), .95 - 1e-13)

    def test_joint_event_implies_bound_and_interval(self):
        generator = np.random.default_rng(31)
        for _ in range(1000):
            p = generator.random()
            e0, e1 = generator.uniform(0, .3, size=2)
            q = observed_probability(p, e0, e1)
            n = 8192
            k, k0, k1 = generator.binomial(n, [q, e0, e1])
            r = hoeffding_radius(n)
            if max(abs(k/n-q), abs(k0/n-e0), abs(k1/n-e1)) <= r:
                result = confidence_intervals(k, n, k0, k1, n)
                self.assertLessEqual(result['lower'], p + 1e-14)
                self.assertGreaterEqual(result['upper'], p - 1e-14)
                est, gap = estimate_from_counts(k, n, k0, k1, n)
                self.assertLessEqual(abs(est-p), 2*r/abs(gap) + 1e-14)

    def test_near_singular_correction_increases_variance(self):
        p = .5
        q = observed_probability(p, .45, .45)
        self.assertEqual(q, p)
        raw_variance = q*(1-q)/8192
        corrected_variance = raw_variance/(1-.45-.45)**2
        self.assertAlmostEqual(corrected_variance/raw_variance, 100)

    def test_drift_bias_formula(self):
        p, c0, c1, v0, v1 = .32244470030643, .07, .12, .13, .10
        bias = inverse(observed_probability(p, v0, v1), c0, c1) - p
        expected = ((1-p)*(v0-c0)-p*(v1-c1))/(1-c0-c1)
        self.assertAlmostEqual(bias, expected)
        self.assertGreater(bias, .05)

    def test_invalid_inputs(self):
        for n in [0, -1, True, 3.5]:
            with self.assertRaises(ValueError):
                confidence_intervals(0, n, 0, 0, 10)
        for k in [-1, 1.5, 11, float('nan')]:
            with self.assertRaises(ValueError):
                confidence_intervals(k, 10, 0, 0, 10)
        with self.assertRaises(ValueError):
            observed_probability(-.01, .07, .12)
        with self.assertRaises(ValueError):
            hoeffding_radius(10, alpha=1)


if __name__ == '__main__':
    unittest.main()
