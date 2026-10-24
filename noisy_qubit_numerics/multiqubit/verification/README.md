# Verification — 24 October 2026

`tests.txt` records 18 passing multi-qubit unit tests. The introductory example's
20 tests are separate. All source files were also parsed/compiled; no standalone
Python lint/type-check tool was installed, so those checks are not claimed.

`device_checks.json` records all 27 circuits, exact CZ count multiplication,
unchanged non-CZ operations, ideal fold invariance and compact register sizes.
A finite-shot Aer check uses 131072 shots. The frozen 38 quantum channels and 8
readout channels were independently compared with the source calibration.

Independent review rebuilt the exact Hamiltonian with Kronecker products and
scipy.expm (maximum probability difference 3.50e-15), applied all directional
readout matrices independently (81 fold distributions matched), reconstructed
all estimates from seeds/count sampling (max difference 5.55e-16), and recomputed
all per-case/pool RMSEs and paired-bootstrap summaries. A fresh 2000-bootstrap
calculation shifted interval endpoints by at most 8.64e-5, without changing the
reported scientific conclusions. Saved population means and report tables were
checked against the JSON. Reported improvement is conditional on this snapshot.

The updated report retains the 38 quadratic-vs-linear failures, 12 quadratic-vs-
readout-only failures and secondary continuous-target reversals. The poorly
calibrated shared physical qubit 1 is explicitly identified. Failed/undefined
estimator handling is implemented; none occurred in this saved experiment.

`plot_review.json` documents inspection of all three figures. The PDF is rendered
and visually checked separately. `package_check.json` records the final extracted
archive checks, including tests, statistical regeneration and figures.
