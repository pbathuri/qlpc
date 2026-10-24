# Provenance and scope

I prepared this example on 24 October 2026.

I started from my QLPC function `readout_mitigation_demo` in
[qlpc/mitigation.py](https://github.com/pbathuri/qlpc/blob/63c325baaf22fdcd5c09707963455fa37c78fc15/qlpc/mitigation.py),
commit `63c325baaf22fdcd5c09707963455fa37c78fc15`.
The source repository is private; this package is self-contained.

The retained baseline is Ry(1.2), depolarization rho -> 0.98 rho + 0.02 I/2,
and asymmetric readout errors e0=0.07 and e1=0.12. QLPC called the M3 package.
Here I solve the one-qubit assignment equation explicitly so the inverse,
conditioning and uncertainty propagation can be inspected without that
additional dependency. This is not a reimplementation or benchmark of M3's
large-system matrix-free algorithm. The original program's rounded output
is not used as numerical ground truth.

I added the Lindblad/Bloch formulations,
Euler/RK4/Strang comparisons, positivity and modal-stability diagnostics,
finite-calibration confidence sets, equal-budget Monte Carlo comparisons,
explicit ill-conditioning/drift cases, and the independent Qiskit checks.
These are demonstrations of established methods, not novelty claims.

The dynamical parameters Omega=0.6, Delta=0.2 rad/us, T1=50us and T2=40us are
synthetic. I chose them to expose numerical behaviour, not to fit a device.
The two-state calibration preparations are ideal by assumption.

Core validation used Python 3.13.13, NumPy 2.4.5, SciPy 1.17.1 and Matplotlib
3.11.0. The separate Qiskit check used Python 3.13.13, NumPy 2.5.4, SciPy 1.18.1,
Qiskit 2.6.0 and Aer 0.18.0. The main results are reproducible from the included
code, configuration and seeds. Last-bit differences can occur on other stacks.
