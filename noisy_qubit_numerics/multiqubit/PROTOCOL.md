# Fixed experimental protocol — 24 October 2026

Specified before inspecting mitigation outcomes. All cases and failed estimates
will remain in the saved results and report. No hardware job is submitted.

- Model: open transverse-field Ising chain, H = sum Z_j Z_(j+1) + 0.7 sum X_j,
  initial state all zero; dimensionless J=1 sets the time unit.
- Register sizes: 4, 6 and 8 qubits. Evolution times: 0.5, 1.0 and 1.5 (1/J).
  Second-order product-formula steps: 2, 4 and 8. Total: 27 circuit cases.
- Primary observables: mean Z magnetization and mean nearest-neighbor ZZ.
  Judge mitigation against the ideal digital circuit. Separately report its
  product-formula bias against exact Hamiltonian evolution.
- Noise: dated ibm_kingston calibration bundled with FakeKingston. Save selected
  physical qubits, native gate errors/durations, T1/T2, readout parameters, package
  versions and the frozen remapped Aer model. Use one valid connected eight-qubit
  path, selected from calibration/topology before seeing circuit outcomes; use
  prefixes for the smaller cases. Exclude unusable calibration entries, not
  circuits with poor mitigation results.
- Amplification: native CZ-only identity folding at factors 1, 3, 5. Barriers
  prevent cancellation; folded CZ counts must scale exactly and ideal states
  must agree. One-qubit noise is not amplified: the intercept is not a claim of
  removing all physical noise. Fixed linear least-squares and quadratic fits.
- Protocols: raw; factorized readout correction; linear ZNE alone; readout plus
  linear ZNE; readout plus quadratic ZNE. No result-dependent fit selection.
- Budget: 32,768 total shots per observable-sharing experiment, including two
  calibration circuits of 2,048 shots each whenever readout correction is used.
  Split remaining measurement shots as equally as possible over required folds.
  Raw uses all shots at fold 1. Report CZ gate-shots as a separate resource cost.
- Calibration circuits prepare all-zero/all-one states using the same gate noise;
  calibration includes preparation error. The factorized inverse assumes local
  readout errors, matching the Aer model; preparation error can bias that inverse.
- Sampling: 500 independent synthetic shot/calibration replicates per case and
  method from saved exact noisy density-matrix distributions. Use fixed seeds;
  do not describe these as hardware repetitions. Save all trial estimates.
- Report bias, RMSE, paired RMSE ratios, out-of-range estimates and undefined
  inversions. Bootstrap replicate rows for uncertainty on aggregate RMSE ratios.
  Sampling uncertainty does not include mismatch of the device noise model.
- Validate calibration remapping, ideal fold invariance, bit ordering, observable
  weights, budget arithmetic, readout inversion, fit coefficients and at least
  one finite-shot Aer sample against the saved distribution.

## Estimator and comparison details (fixed before outcomes)

Linear intercept weights at (1,3,5): (13/12,1/3,-5/12); quadratic:
(15/8,-5/4,3/8). Each method receives the stated full shot budget.
Within one replicate, nested shot prefixes couple methods that sample the
same fold; readout-corrected methods share that replicate's calibration draw,
and both corrected ZNE fits share fold counts. This common-random-number
coupling is only for Monte Carlo comparison; it does not waive any protocol's
calibration cost. Replicate rows and circuit cases are independent.

Aggregate RMSE is sqrt(equal-weight mean squared error over all 27 cases,
both observables, and all replicate rows). Paired bootstrap resamples rows
within each fixed case, retaining all method/observable/calibration dependence;
its 95% intervals measure conditional Monte Carlo uncertainty, not uncertainty
in the noise model or an ensemble of unspecified devices. They are not
simultaneous per-circuit guarantees. If a method has any undefined inversion,
its unqualified aggregate is undefined; finite-only summaries are labeled.

A CZ-noise-disabled diagnostic, if computed, is not a fitting point. It keeps
one-qubit and readout noise to expose the residual unscaled-noise floor.
