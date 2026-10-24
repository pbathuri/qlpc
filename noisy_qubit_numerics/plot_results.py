"""Three plain figures from saved results; no simulation or source-data changes."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
COLORS = {'euler': '#D55E00', 'rk4': '#0072B2', 'strang': '#009E73'}
NAMES = {'euler': 'Forward Euler', 'rk4': 'RK4', 'strang': 'Strang (exact substeps)'}


def finish(fig, filename, date):
    fig.text(.5, .015, f'Source: data/results.json; {date}; synthetic qubit simulation, not hardware measurements.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .97))
    fig.savefig(ROOT/'figures'/filename, dpi=180)
    plt.close(fig)


def style(ax):
    ax.grid(axis='y', which='major', color='.85', linewidth=.5)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    result = json.loads((ROOT/'data/results.json').read_text())
    (ROOT/'figures').mkdir(exist_ok=True)
    d = result['dynamics']
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(d['times_us'], d['ideal_p1_trajectory'], '--', color='.35', label='Noiseless reference')
    ax.plot(d['times_us'], d['reference_p1_trajectory'], '-', color='black', label='Lindblad matrix-exponential reference')
    for method, marker in [('euler', 'x'), ('strang', 'o')]:
        values = d['coarse_trajectories'][method]
        ax.plot(values['times_us'], values['p1'], marker+'-', markersize=4,
                color=COLORS[method], label=NAMES[method]+'; h = 1 us')
    ax.axhline(0, color='.6', linewidth=.6)
    ax.axhline(1, color='.6', linewidth=.6)
    ax.set(title='Driven-qubit excited-state population: physical noise and time-stepping error',
           xlabel='Evolution time (microseconds)', ylabel='Excited-state population p1 (unitless)')
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc='best')
    finish(fig, 'dynamics.png', result['date'])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for method in ['euler', 'rk4', 'strang']:
        rows = [r for r in d['convergence'] if r['method'] == method]
        steps = [r['steps'] for r in rows]
        axes[0].loglog(steps, [r['terminal_half_trace_norm_error'] for r in rows],
                       'o-', color=COLORS[method], label=NAMES[method])
        axes[1].plot(steps, [r['minimum_eigenvalue_over_trajectory'] for r in rows],
                     {'euler': 'o-', 'rk4': 'x-', 'strang': 's--'}[method], color=COLORS[method],
                     markersize=5, label=NAMES[method])
    axes[0].set(title='Convergence at 10 microseconds', xlabel='Time steps (count)',
                ylabel='Half trace-norm error versus reference (unitless)')
    axes[1].set_xscale('log', base=2)
    axes[1].set_yscale('symlog', linthresh=1e-4)
    axes[1].set_ylim(1.35*min(r['minimum_eigenvalue_over_trajectory'] for r in d['convergence']), 2e-4)
    axes[1].axhline(0, color='.5', linewidth=.6)
    axes[1].set(title='Physicality over every saved time step', xlabel='Time steps (count)',
                ylabel='Minimum density-matrix eigenvalue (unitless)')
    axes[1].text(.98, .18, 'Negative = nonphysical\nRK4 and Strang overlap at 0',
                 ha='right', transform=axes[1].transAxes, fontsize=8)
    for ax in axes:
        style(ax)
        ax.legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8, loc='lower right')
    finish(fig, 'convergence.png', result['date'])

    cases = result['readout']['cases']
    positions = np.arange(len(cases))
    labels = ['Driven\nqubit', 'QLPC\nbaseline', 'Balanced\nill-conditioned', 'Boundary\nill-conditioned', 'Drifted\ncalibration']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = .24
    for offset, field, label, color in [(-width, 'raw', 'Raw: 8,192 shots', '.6'),
                                      (0, 'raw_matched_total_budget', 'Raw: 24,576 shots', '#0072B2'),
                                      (width, 'estimated_calibration', 'Corrected: 24,576 total shots', '#D55E00')]:
        axes[0].plot(positions+offset, [c[field]['rmse_finite'] for c in cases], 'o', color=color,
                     markersize=6, label=label)
    axes[0].set_yscale('log')
    axes[0].set(title='Correction can increase total estimation error',
                ylabel='RMSE versus pre-readout population (unitless)', xlabel='Simulation case')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    axes[1].bar(positions, [100*c['nominal_interval_coverage'] for c in cases],
                color=['#0072B2']*4+['#D55E00'], width=.6, label='Observed coverage of nominal 95% sets')
    axes[1].axhline(95, color='.25', linestyle='--', linewidth=1, label='95% lower guarantee if model holds')
    for i, c in enumerate(cases):
        axes[1].text(i, 100*c['nominal_interval_coverage']+1.5,
                     f"{100*c['nominal_interval_coverage']:g}%", ha='center', fontsize=9)
    axes[1].set(ylim=(0, 112), title='Calibration drift invalidates the confidence guarantee',
                ylabel='Coverage over 5,000 trials (%)', xlabel='Simulation case')
    axes[1].legend(frameon=False, fontsize=8, loc='lower left')
    for ax in axes:
        ax.set_xticks(positions, labels, fontsize=8)
        style(ax)
    finish(fig, 'mitigation.png', result['date'])
    print('Wrote figures/dynamics.png, convergence.png, mitigation.png')


if __name__ == '__main__':
    main()
