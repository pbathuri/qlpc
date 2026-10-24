# Device-calibrated multi-qubit benchmark

I compare mitigation methods on entangling 4-, 6- and 8-qubit Ising circuits.
The [main write-up](../WRITEUP.pdf) explains the results and limitations.
All observations are generated in Aer from the 15 April 2026 ibm_kingston
calibration bundled with FakeKingston; none are live hardware measurements.

## Reproduce

Python 3.13 is the tested interpreter. From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -p 'test_estimators.py'
python run_benchmark.py
python plot_benchmark.py
```

To regenerate the calibrated quantum-circuit distributions, use a separate
simulator environment from that same directory:

```sh
python3 -m venv .venv-simulator
.venv-simulator/bin/python -m pip install -r requirements-simulator.txt
.venv-simulator/bin/python -m unittest discover -p 'test_device_circuits.py'
.venv-simulator/bin/python generate_distributions.py
.venv/bin/python run_benchmark.py
.venv/bin/python plot_benchmark.py
```

Together, the two test commands run 18 tests covering calibration remapping,
channel equivalence, gate folding,
bit ordering, observable inversion, estimator coefficients and shot budgets.
Generation takes seconds to minutes on a laptop; no HPC or service login is
needed. It overwrites generated results in this directory, so copy them first
if you want to compare experiments. Do not use `--refresh-device` unless you
intend to replace the frozen snapshot with the one installed in your runtime.
The device path and model are frozen in the included files.

Saved distributions were generated with NumPy 2.5.4/SciPy 1.18.1, Qiskit 2.6.0,
Aer 0.18.0 and IBM Runtime 0.50.0. Recorded statistical analysis used
NumPy 2.4.5/SciPy 1.17.1; plotting used Matplotlib 3.11.0. Exact package versions
and hashes are recorded. NumPy 2.4 and 2.5 have different multinomial sampling
streams: use the pinned analysis environment to reproduce saved shot trials.
The two requirement files intentionally keep simulator and analysis environments
separate. Floating-point last bits can also vary by platform.

## Files and methods

- `device_circuits.py`: frozen calibration, compact native circuits, CZ folding,
  exact Hamiltonian reference, Aer channels and explicit directional readout.
- `generate_distributions.py`: all 27 fixed cases and a finite-shot Aer check.
- `estimators.py`: independent assignment inversion, fixed extrapolation weights,
  and nested multinomial shot prefixes.
- `run_benchmark.py`: equal-budget resampling, paired bootstrap and full reports.
- `plot_benchmark.py`: three figures generated solely from saved results.
- `PROTOCOL.md`: design fixed before inspecting mitigation outcomes.
- `data/circuits.json`: physical layout, native counts and calibration details.
- `data/device_noise.json` and `device_calibration.json`: frozen channel model,
  directional readout properties, calibration timestamps and source hashes.
- `data/distributions.npz`: ideal digital, continuous-Hamiltonian, noisy folded,
  pre-readout and CZ-noise-disabled diagnostic distributions (little-endian).
- `data/benchmark.json` and `all_results.csv`: every method/case/observable result.
- `data/trials.npz`: every replicate estimate and calibration marginal count.
  Measurement counts are regenerated from saved distributions and fixed seeds.

The primary target is the ideal digital circuit. Exact Hamiltonian evolution
is a distinct secondary reference. Both are recorded. The circuit grid covers
n=4,6,8; T=.5,1,1.5 in units 1/J; and 2,4,8 second-order product-formula steps.
No cases, seeds, fit degrees or noise factors are selected for good results.

## Fairness and inference

Each protocol gets 32,768 shots. With readout correction, 2,048 shots prepare each
of all-zero and all-one; preparation X gates carry the same device noise.
The remaining 28,672 shots are used on fold 1 or split 9558/9557/9557 over folds 1/3/5.
Uncorrected ZNE splits 32768 shots as 10923/10923/10922. Both observables share
bitstrings. ZNE costs about 3 times raw CZ gate-shots (2.625 times with readout
calibration); it is not an equal-gate-cost experiment.

Common-random-number shot prefixes and shared within-replicate calibration
couple method comparisons. This couples comparisons without changing each method's marginal sampling
distribution; each method is charged its full counterfactual budget. Bootstrap resampling
preserves row dependence and treats the fixed cases separately. Intervals
are conditional sampling intervals, not device-model uncertainty or simultaneous
guarantees. The quadratic fit has zero residual degrees of freedom.

Pooled RMSE weights all 27 cases and both observables equally. Any undefined
inversion makes a full aggregate undefined; finite-only statistics are labeled.
Finite estimates outside [-1,1] are retained. None occurred in the saved runs.

## Important limitations

The topology-first physical path 0..7 contains qubit 1 with substantial asymmetric
readout error, and every smaller register includes it. Thus all 54 comparisons
share that condition. Report incremental gains over readout-only, not only raw.
CZ-only folding does not scale one-qubit/preparation/readout noise. The CZ-noise-off
reference keeps those errors and is excluded from fits. The model does not
include timed idle evolution, crosstalk, leakage, correlated readout or drift.
Density-matrix calculations are accurate for this chosen model, not a claim
that the model completely describes the real device.

M3 and IBM's hardware studies are literature context; this program uses explicit
factorized readout inversion and native CZ folding. It does not execute M3,
reproduce a large-scale hardware result, or establish a new quantum advantage.
