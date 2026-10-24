"""Transparent one-qubit readout inversion, adapted from QLPC's noise example.

For e0=P(report 1|0), e1=P(report 0|1), q=e0+(1-e0-e1)*p.
Correction targets the pre-measurement population, not the noiseless circuit.
No estimate is clipped to hide inversion failures. Calibration preparations
are assumed ideal, and the confidence guarantee requires stable readout rates.
"""

from itertools import product

import numpy as np


def observed_probability(p, e0, e1):
    if not (0 <= e0 <= 1 and 0 <= e1 <= 1):
        raise ValueError("Readout errors must lie in [0,1].")
    p = np.asarray(p, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Sampling requires a physical probability.")
    return e0 + (1 - e0 - e1) * p


def inverse(q, e0, e1):
    """Unconstrained inverse; an exactly singular channel raises an error."""
    gap = 1 - e0 - e1
    if gap == 0:
        raise ValueError("Readout assignment is singular.")
    return (np.asarray(q) - e0) / gap


def _counts(counts, shots):
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError("Shots must be a positive integer.")
    values = np.asarray(counts, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values != np.floor(values)):
        raise ValueError("Counts must be finite integers.")
    if np.any((values < 0) | (values > shots)):
        raise ValueError("Counts must lie between zero and shots.")
    return values


def estimate_from_counts(counts, shots, cal0, cal1, calibration_shots):
    q = _counts(counts, shots) / shots
    k0 = _counts(cal0, calibration_shots)
    k1 = _counts(cal1, calibration_shots)
    # Use the integer numerator to detect singularity exactly, not by tolerance.
    gap = (calibration_shots - k0 - k1) / calibration_shots
    numerator = q - k0 / calibration_shots
    q, gap, numerator = np.broadcast_arrays(q, gap, numerator)
    estimate = np.full(q.shape, np.nan)
    np.divide(numerator, gap, out=estimate, where=gap != 0)
    return estimate, gap


def hoeffding_radius(shots, alpha=0.05, comparisons=3):
    _counts(0, shots)
    if not 0 < alpha < 1 or comparisons < 1:
        raise ValueError("Invalid confidence level or comparison count.")
    return float(np.sqrt(np.log(2 * comparisons / alpha) / (2 * shots)))


def invert_box(q_lo, q_hi, e0_lo, e0_hi, e1_lo, e1_hi):
    """Enclose the inverse over a joint box using all eight vertices.

    A box not wholly within positive assignment gaps returns flagged [0,1]. An empty physical
    intersection is retained (lower>upper), so coverage checks cannot hide it.
    """
    arrays = np.broadcast_arrays(q_lo, q_hi, e0_lo, e0_hi, e1_lo, e1_hi)
    q_lo, q_hi, e0_lo, e0_hi, e1_lo, e1_hi = arrays
    for low, high in ((q_lo, q_hi), (e0_lo, e0_hi), (e1_lo, e1_hi)):
        if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
            raise ValueError("Box endpoints must be finite.")
        if np.any((low < 0) | (high > 1) | (low > high)):
            raise ValueError("Invalid probability box.")
    nonsingular = 1 - e0_hi - e1_hi > 0
    vertices = []
    for q, e0, e1 in product((q_lo, q_hi), (e0_lo, e0_hi), (e1_lo, e1_hi)):
        denominator = np.where(nonsingular, 1 - e0 - e1, 1.0)
        vertices.append((q - e0) / denominator)
    lower = np.where(nonsingular, np.maximum(0, np.min(vertices, axis=0)), 0.0)
    upper = np.where(nonsingular, np.minimum(1, np.max(vertices, axis=0)), 1.0)
    return {"lower": lower, "upper": upper, "positive_gap_not_certified": ~nonsingular,
            "empty": lower > upper}


def confidence_intervals(counts, shots, cal0, cal1, calibration_shots, alpha=0.05):
    """At least 1-alpha coverage for a fixed p under the stable Bernoulli model.

    Three two-sided Hoeffding bounds use alpha/3 each, then a union bound.
    This is conservative; neither a trajectory-wide band nor a drift bound.
    """
    q = _counts(counts, shots) / shots
    e0 = _counts(cal0, calibration_shots) / calibration_shots
    e1 = _counts(cal1, calibration_shots) / calibration_shots
    rq = hoeffding_radius(shots, alpha)
    rc = hoeffding_radius(calibration_shots, alpha)
    bounds = invert_box(np.maximum(0, q-rq), np.minimum(1, q+rq),
                        np.maximum(0, e0-rc), np.minimum(1, e0+rc),
                        np.maximum(0, e1-rc), np.minimum(1, e1+rc))
    bounds.update(measurement_radius=rq, calibration_radius=rc)
    return bounds


def known_rate_interval(counts, shots, e0, e1, alpha=0.05):
    if not (0 <= e0 <= 1 and 0 <= e1 <= 1 and e0 + e1 < 1):
        raise ValueError("Known-rate interval needs a positive assignment gap.")
    q = _counts(counts, shots) / shots
    r = hoeffding_radius(shots, alpha, comparisons=1)
    return invert_box(np.maximum(0, q-r), np.minimum(1, q+r), e0, e0, e1, e1)
