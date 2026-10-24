# Verification — 24 October 2026

The core unit suite passes 20 tests; stdout/stderr are in tests.txt. All seven
Python files parse and compile. No Python lint/type checker was installed in
the validation environment; those checks are not claimed.

Independent read-only review reconstructed all 20 saved count arrays from
seeds, recomputed biases, RMSEs, outside-[0,1] counts, and confidence coverage,
and reproduced all 24 convergence rows. There were no discrepancies. The
report's numerical values were checked against results.json and the Qiskit
record. The main references share scipy.linalg.expm but are assembled in
independent coordinates; DOP853 supplies a separate algorithmic comparison.

qiskit_check.json records the optional check actually running with Qiskit2.6.0
and Aer0.18.0. It includes 138 channel-state comparisons and an independent
200,000-shot reconstruction of the QLPC forward-noise experiment. This checks
simulation consistency, not real hardware and not the original M3 output.

Theoretical review checked the dephasing convention, all-eight-corner inverse
propagation, nonzero-denominator condition, equal-total-shot accounting,
positivity diagnostics, finite-calibration ratio bias, and the distinction
between stable-model concentration and deliberate calibration drift.

All three figures and all three PDF pages were visually inspected for labels,
units, clipping and mathematical typesetting. All noisy figures explicitly
say synthetic simulation. The write-up does not claim hardware validation,
quantum error correction, novelty, scalability, or computational advantage.

package_check.json records testing from an extracted ZIP: unit tests, complete
experiment regeneration, figure regeneration and optional Qiskit validation.
It also compares regenerated JSON and all trial arrays with the shipped data.
SHA256SUMS.txt at the package root covers every included payload file.
