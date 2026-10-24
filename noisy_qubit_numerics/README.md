# Noisy-qubit numerical methods

I study how numerical approximation, device noise and statistical correction
interact in quantum circuits. Start with the updated [three-page note](WRITEUP.pdf)
and the [multi-qubit study](multiqubit/). The earlier one-qubit example remains
available as a smaller introduction.

## Multi-qubit study

I simulate 4-, 6- and 8-qubit transverse-field Ising circuits using a frozen
IBM Kingston calibration dated **15 April 2026**. I compare raw estimates,
factorized readout correction, and linear/quadratic CZ-fold zero-noise
extrapolation at **32,768 total shots per protocol**, including calibration.
These are calibrated Aer simulations, **not live hardware runs**.

The fixed grid has 27 circuits and two observables, with 500 synthetic shot
replicates per circuit. All cases are retained. The combined methods lower
pooled error against the ideal digital circuit, but quadratic extrapolation
is worse than linear in 38 of 54 comparisons. Accuracy against exact
Hamiltonian evolution is reported separately: correcting noise does not
remove product-formula error or the possibility of error cancellation.

- [Methods, data and run instructions](multiqubit/README.md)
- [Fixed protocol](multiqubit/PROTOCOL.md)
- [Three figures](multiqubit/figures/)
- [All 270 comparison rows](multiqubit/data/all_results.csv)
- [Frozen calibration and results](multiqubit/data/)
- [Verification record](multiqubit/verification/README.md)

## Run the main study

From this folder, with Python 3.13:

```sh
cd multiqubit
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

The generator reuses the included frozen model by default. No IBM account,
external dataset, cluster allocation or hardware submission is needed. The
commands replace generated files in multiqubit/data/, figures/ and verification/.
Use the included saved outputs to read the results without installing anything.

## Earlier one-qubit introduction

[SINGLE_QUBIT_WRITEUP.pdf](SINGLE_QUBIT_WRITEUP.pdf) and
[SINGLE_QUBIT_README.md](SINGLE_QUBIT_README.md) document the original Lindblad,
positivity and uncertain-readout examples. Their programs and 20 unit tests
remain at this folder's top level. Their synthetic parameters are separate
from the device-derived multi-qubit experiment.

## Scope

The readout model is independent per qubit, and CZ folding amplifies only
CZ noise. Equal shots are not equal gate execution cost. The study does not
model a timed idle schedule, crosstalk, leakage, correlated readout or changes
in calibration. It is a reproducible small-system benchmark of established
methods, not a new mitigation algorithm, hardware claim or quantum advantage.

From the package root, rebuild the updated note with Pandoc and XeLaTeX:

```sh
pandoc WRITEUP.md --pdf-engine=xelatex -o WRITEUP.pdf
```
