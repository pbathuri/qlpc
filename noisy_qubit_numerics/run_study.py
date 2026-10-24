"""Reproduce every numerical experiment; writes data/ within this standalone folder."""

from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from scipy.integrate import solve_ivp

from dynamics import Model, integrate, liouvillian, p1, reference_bloch, reference_expm, trace_error
from readout import confidence_intervals, estimate_from_counts, hoeffding_radius, inverse, observed_probability

ROOT = Path(__file__).resolve().parent
DATE = '2026-10-24'
SEED = 20261024


def diagnostics(trajectory):
    return {
        'minimum_eigenvalue_over_trajectory': float(np.linalg.eigvalsh(trajectory).min()),
        'maximum_trace_error': float(np.max(np.abs(np.trace(trajectory, axis1=1, axis2=2)-1))),
        'maximum_hermiticity_error': float(np.max(np.abs(trajectory-trajectory.conj().transpose(0, 2, 1)))),
    }


def dynamics_experiment():
    model = Model()
    duration = 10.0
    steps_list = [5, 10, 20, 40, 80, 160, 320, 640]
    reference = reference_expm(model, duration)
    times = np.linspace(0, duration, 201)
    references = np.array([reference_expm(model, t) for t in times])
    ideal_model = Model(model.omega, model.detuning, np.inf, np.inf)
    ideal = np.array([p1(reference_expm(ideal_model, t)) for t in times])
    bloch_error = max(trace_error(rho, reference_bloch(model, t)) for rho, t in zip(references, times))
    generator = liouvillian(model)
    modes = np.linalg.eigvals(generator)
    decaying_modes = modes[np.abs(modes) > 1e-12]
    euler_step_limit = float(np.min(-2*decaying_modes.real/np.abs(decaying_modes)**2))
    initial = reference_expm(model, 0).reshape(4, order='F')
    ode = solve_ivp(lambda t, y: generator @ y, (0, duration), initial,
                    method='DOP853', rtol=1e-12, atol=1e-14)
    if not ode.success:
        raise RuntimeError(ode.message)
    ode_error = trace_error(ode.y[:, -1].reshape(2, 2, order='F'), reference)
    rows = []
    coarse = {}
    for method in ['euler', 'rk4', 'strang']:
        previous = None
        for steps in steps_list:
            trajectory = integrate(model, duration, steps, method)
            error = trace_error(trajectory[-1], reference)
            row = {'method': method, 'steps': steps, 'step_us': duration/steps,
                   'terminal_half_trace_norm_error': error,
                   'terminal_p1': p1(trajectory[-1]), **diagnostics(trajectory)}
            if method == 'euler':
                row['decaying_mode_spectral_radius'] = float(np.max(np.abs(1+duration/steps*decaying_modes)))
            row['observed_order'] = None if previous is None else float(np.log2(previous/error))
            rows.append(row)
            previous = error
            if steps == 10:
                coarse[method] = {'times_us': np.linspace(0, duration, steps+1).tolist(),
                                  'p1': [p1(rho) for rho in trajectory]}
    return {'model': asdict(model), 'duration_us': duration, 'steps': steps_list,
            'reference_p1': p1(reference), 'ideal_p1': float(ideal[-1]),
            'physical_noise_shift': p1(reference)-float(ideal[-1]),
            'decaying_generator_eigenvalues_real_imag': [[float(z.real), float(z.imag)] for z in decaying_modes],
            'euler_linear_stability_step_limit_us': euler_step_limit,
            'bloch_max_reference_discrepancy': bloch_error,
            'dop853_terminal_reference_discrepancy': ode_error,
            'reference_diagnostics': diagnostics(references),
            'times_us': times.tolist(), 'reference_p1_trajectory': [p1(rho) for rho in references],
            'ideal_p1_trajectory': ideal.tolist(), 'coarse_trajectories': coarse, 'convergence': rows}


def summary(values, target):
    finite = np.isfinite(values)
    errors = values[finite]-target
    return {'finite_count': int(finite.sum()), 'undefined_count': int((~finite).sum()),
            'bias_finite': float(errors.mean()) if finite.any() else None,
            'rmse_finite': float(np.sqrt(np.mean(errors**2))) if finite.any() else None,
            'outside_unit_interval_count': int(np.sum(finite & ((values < 0) | (values > 1))))}


def readout_experiment(driven_p):
    ideal = float(np.sin(1.2/2)**2)
    noisy = .98*ideal + .01
    repetitions, shots, calibration_shots, alpha = 5000, 8192, 8192, .05
    definitions = [
        ('Driven qubit', driven_p, (.07, .12), (.07, .12)),
        ('QLPC baseline', noisy, (.07, .12), (.07, .12)),
        ('Balanced, ill-conditioned', .5, (.45, .45), (.45, .45)),
        ('Boundary, ill-conditioned', .02, (.45, .45), (.45, .45)),
        ('Drifted calibration', noisy, (.07, .12), (.13, .10)),
    ]
    cases = []
    saved = {}
    for index, (name, target, calibration, actual) in enumerate(definitions):
        rng = np.random.default_rng(SEED+index)
        c0, c1 = calibration
        e0, e1 = actual
        observed = float(observed_probability(target, e0, e1))
        counts = rng.binomial(shots, observed, repetitions)
        cal0 = rng.binomial(calibration_shots, c0, repetitions)
        cal1 = rng.binomial(calibration_shots, c1, repetitions)
        total_shots = shots + 2*calibration_shots
        matched = rng.binomial(total_shots, observed, repetitions)
        estimated, gaps = estimate_from_counts(counts, shots, cal0, cal1, calibration_shots)
        known = inverse(counts/shots, e0, e1)
        intervals = confidence_intervals(counts, shots, cal0, cal1, calibration_shots, alpha)
        coverage = (intervals['lower'] <= target) & (target <= intervals['upper'])
        finite = np.isfinite(estimated)
        width = np.maximum(0, intervals['upper']-intervals['lower'])
        drift_bias = ((1-target)*(e0-c0)-target*(e1-c1))/(1-c0-c1)
        cases.append({
            'name': name, 'seed': SEED+index, 'target_before_readout': target,
            'actual_e0': e0, 'actual_e1': e1, 'calibration_e0': c0, 'calibration_e1': c1,
            'readout_expected_q': observed, 'actual_gap': 1-e0-e1,
            'stable_calibration_model': calibration == actual,
            'population_bias_with_stale_calibration': drift_bias,
            'known_rate_variance_amplification': 1/(1-e0-e1)**2,
            'raw': summary(counts/shots, target),
            'raw_matched_total_budget': summary(matched/total_shots, target),
            'known_current_rates_oracle': summary(known, target),
            'estimated_calibration': summary(estimated, target),
            'negative_estimated_gap_count': int(np.sum(gaps < 0)),
            'zero_estimated_gap_count': int(np.sum(gaps == 0)),
            'nominal_interval_coverage': float(coverage.mean()),
            'covered_trials': int(coverage.sum()),
            'mean_interval_width': float(width.mean()),
            'positive_gap_not_certified_fraction': float(np.mean(intervals['positive_gap_not_certified'])),
            'empty_interval_count': int(np.sum(intervals['empty'])),
            'median_point_error_bar_finite': float(np.median(2*hoeffding_radius(shots)/np.abs(gaps[finite]))),
        })
        for key, value in {'measurement': counts, 'cal0': cal0, 'cal1': cal1, 'raw_matched': matched}.items():
            saved[f'case{index}_{key}'] = value
    guard = confidence_intervals(4096, 8192, 4014, 4014, 8192)
    return {'repetitions': repetitions, 'measurement_shots': shots,
            'calibration_shots_per_basis_state': calibration_shots,
            'total_shots_per_estimated_calibration_trial': shots+2*calibration_shots,
            'nominal_alpha': alpha, 'hoeffding_radius_each': hoeffding_radius(shots),
            'calibration_preparation': 'ideal basis states; independent fresh calibration each repetition',
            'qlpc_baseline': {'ry_angle': 1.2, 'depolarizing_parameter': .02,
                              'ideal_p1': ideal, 'after_depolarizing_p1': noisy,
                              'after_readout_q': float(observed_probability(noisy, .07, .12)),
                              'residual_gate_bias_after_perfect_readout_correction': noisy-ideal},
            'singular_guard_example': {'measurement_count': 4096, 'cal0_count': 4014, 'cal1_count': 4014,
                                       'lower': float(guard['lower']), 'upper': float(guard['upper']),
                                       'positive_gap_not_certified': bool(guard['positive_gap_not_certified'])},
            'cases': cases}, saved


def main():
    dynamics = dynamics_experiment()
    readout, saved = readout_experiment(dynamics['reference_p1'])
    result = {'date': DATE, 'provenance': 'Adapted from qlpc/mitigation.py at 63c325baaf22fdcd5c09707963455fa37c78fc15',
              'scope': 'Synthetic physically motivated model; simulation only, no hardware calibration or live device',
              'python': platform.python_version(),
              'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'matplotlib']},
              'dynamics': dynamics, 'readout': readout}
    out = ROOT/'data'
    out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(out/'readout_trials.npz', **saved)
    print('Saved data/results.json and all 25,000 trial count records in data/readout_trials.npz.')
    print('Dynamics reference p1:', dynamics['reference_p1'])
    for row in readout['cases']:
        print(row['name'], 'raw RMSE=', row['raw']['rmse_finite'],
              'corrected RMSE=', row['estimated_calibration']['rmse_finite'],
              'nominal coverage=', row['nominal_interval_coverage'])


if __name__ == '__main__':
    main()
