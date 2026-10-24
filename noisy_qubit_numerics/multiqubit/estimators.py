"""Fixed readout and CZ-fold estimators; no circuit or result-dependent tuning."""
import numpy as np

FACTORS = (1, 3, 5)
METHODS = ('raw', 'readout', 'zne_linear', 'readout_zne_linear', 'readout_zne_quadratic')
OBSERVABLES = ('mean_z', 'mean_neighbor_zz')
LINEAR = np.array([13/12, 1/3, -5/12])
QUADRATIC = np.array([15/8, -5/4, 3/8])


def bit_table(n):
    if not isinstance(n, int) or n < 2:
        raise ValueError('At least two qubits are required.')
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)


def observable_weights(n):
    z = 1-2*bit_table(n)
    return np.column_stack((z.mean(axis=1), (z[:, :-1]*z[:, 1:]).mean(axis=1)))


def corrected_weights(e0, e1):
    """Independent local assignment inverse, with no clipping or regularization.

    e0/e1 may be one vector or a batch. Invalid estimated gaps flag an entire
    row as NaN; negative but nonzero gaps are retained, not silently removed.
    """
    e0, e1 = np.broadcast_arrays(np.atleast_2d(e0), np.atleast_2d(e1))
    gap = 1-e0-e1
    valid = np.all(gap != 0, axis=1)
    z = 1-2*bit_table(e0.shape[1])
    with np.errstate(divide='ignore', invalid='ignore'):
        corrected = (z[None, :, :]-(e1-e0)[:, None, :])/gap[:, None, :]
    weights = np.stack((corrected.mean(axis=2),
                        (corrected[:, :, :-1]*corrected[:, :, 1:]).mean(axis=2)), axis=2)
    weights[~valid] = np.nan
    return weights


def calibration_rates(zero_counts, one_counts, shots, n):
    """All-zero/all-one calibration, including any noisy preparation effects."""
    if shots <= 0:
        raise ValueError('Positive calibration shots required.')
    bits = bit_table(n)
    k0 = np.asarray(zero_counts) @ bits
    k1 = shots-np.asarray(one_counts) @ bits
    if np.any(k0 < 0) or np.any(k0 > shots) or np.any(k1 < 0) or np.any(k1 > shots):
        raise ValueError('Invalid calibration counts.')
    return k0/shots, k1/shots, shots-k0-k1


def shot_plan(total=32768, calibration_shots=2048):
    if not isinstance(total, int) or not isinstance(calibration_shots, int) or calibration_shots <= 0:
        raise ValueError('Integer budgets and positive calibration shots required.')
    remaining = total-2*calibration_shots
    if remaining < 3:
        raise ValueError('Budget cannot fund calibration and all three folds.')
    def split(n):
        quotient, remainder = divmod(n, 3)
        return [quotient+(i < remainder) for i in range(3)]
    return {'raw': [total, 0, 0], 'readout': [remaining, 0, 0],
            'zne_linear': split(total), 'readout_zne_linear': split(remaining),
            'readout_zne_quadratic': split(remaining)}


def nested_counts(rng, probability, budgets, repeats):
    """Multinomial shot prefixes for paired, equal-marginal-budget comparisons."""
    p = np.asarray(probability, dtype=float)
    if np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError('Invalid sampling distribution.')
    out = {}
    cumulative = np.zeros((repeats, p.size), dtype=np.int64)
    previous = 0
    for budget in sorted(set(int(n) for n in budgets if n > 0)):
        cumulative = cumulative+rng.multinomial(budget-previous, p, size=repeats)
        out[budget] = cumulative.copy()
        previous = budget
    return out


def estimate_counts(counts, weights):
    shots = np.asarray(counts).sum(axis=1)
    if np.any(shots <= 0):
        raise ValueError('Empty measurement sample.')
    if np.asarray(weights).ndim == 2:
        return np.asarray(counts) @ weights / shots[:, None]
    return np.einsum('rs,rso->ro', counts, weights)/shots[:, None]


def extrapolate(values, degree):
    if degree not in (1, 2):
        raise ValueError('Only the fixed linear and quadratic fits are defined.')
    if np.asarray(values).shape[0] != 3:
        raise ValueError('All three fixed fold factors are required.')
    return np.tensordot(LINEAR if degree == 1 else QUADRATIC, values, axes=(0, 0))
