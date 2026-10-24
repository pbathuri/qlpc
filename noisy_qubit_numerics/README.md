# Noisy-qubit numerical methods

I built this example to explore numerical stability and error propagation
in noisy quantum systems.
Start with **WRITEUP.pdf** (short note), then **run_study.py**.

I compare numerical integration of a one-qubit Lindblad equation with
readout-error mitigation and retain the failure cases: Euler can create nonphysical
states, inversion can amplify sampling noise, and calibration drift can
invalidate confidence sets. All data are simulated with synthetic parameters;
there are no live-device measurements or hardware-calibration claims.

## Run the self-contained example

Python 3.13 was used for the recorded results. No account, external data or
quantum SDK is needed for the core study.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -p 'test_*.py'
python run_study.py
python plot_results.py
```

The commands regenerate the two files in data/ and the three PNGs in figures/.
They run quickly on a laptop. The test suite has 20 tests and does not silently
fall back to a different algorithm. Package versions and seeds are recorded in
data/results.json. A requirements file with tested versions is included.

## What to read

- dynamics.py: Lindblad generator, independent Bloch formulation, Euler, RK4,
  and exact-channel Strang splitting.
- readout.py: the two-state inverse and Hoeffding confidence-box propagation.
- run_study.py: fixed experiment definitions, diagnostics and saved trial counts.
- plot_results.py: plots from saved JSON, without rerunning the experiment.
- test_dynamics.py / test_readout.py: analytic checks and numerical failure cases.
- verification/: recorded tests, optional channel validation and audit results.
- PROVENANCE.md: relationship to the earlier QLPC one-qubit example.

Time is in microseconds; angular frequencies are radians per microsecond.
Probabilities, RMSE and density-matrix eigenvalues are dimensionless.

## Optional independent Qiskit/Aer cross-check

```sh
python -m pip install -r requirements-qiskit.txt
python check_qiskit.py
```

This checks the Strang subchannels using Qiskit density matrices and independently
checks the QLPC depolarization/readout baseline in Aer. It prints JSON and fails
if unavailable or inconsistent; it does not silently substitute classical
results. It does not run M3 or claim to verify the earlier project's M3 output.
The saved successful output is verification/qiskit_check.json.

## How to read the uncertainty results

Each readout case contains 5,000 independent complete trials, each with 8,192
measurement shots and 8,192 calibration shots for each ideal basis state.
Corrections and the raw equal-budget baseline each cost 24,576 shots per trial.
An additional raw 8,192-shot result and a known-current-rate oracle are saved
for diagnostic comparison. The oracle assumes calibration knowledge for free.
Calibration is fresh within each simulated trial, not reused across trials.

The four stable-model cases satisfy the assumptions of the conservative
Hoeffding construction. The deliberate drift case does not. Observed coverage
5000/5000 is not a claim of exact or perfect population coverage. These are
per-experiment intervals, not simultaneous bands over many circuits or times.

Point estimates are never clipped. The results count finite estimates, undefined
zero-denominator estimates, negative estimated gaps and out-of-range estimates.
RMSE fields are named rmse_finite: if undefined estimates occur in a changed
experiment, those fields are conditional on the explicitly reported finite set.
There are no undefined estimates in the recorded five cases. If an interval box
is not wholly within positive assignment gaps, the implementation returns
flagged [0,1]; this flag need not imply the actual channel is singular.
Calibration preparations are assumed ideal. Preparation errors, correlated
shots, non-Markovian noise, leakage and device drift need additional modelling.

## Rebuild the PDF (optional)

With Pandoc and XeLaTeX installed:

```sh
pandoc WRITEUP.md --pdf-engine=xelatex -o WRITEUP.pdf
```

The numerical programs do not require TeX. Figures are shipped separately and
the mitigation figure is also embedded in the note.
