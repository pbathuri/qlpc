#!/bin/bash
# One-time environment setup for the language-conditioned-QAOA study on IU BigRed200 (HPE Cray EX, A100).
# Usage:  bash setup_env.sh
# Adjust the module names if your BigRed200 software stack differs (run `module avail python cuda`).
set -euo pipefail

module load python/gpu 2>/dev/null || module load anaconda 2>/dev/null || true
module load cudatoolkit 2>/dev/null || module load cuda 2>/dev/null || true

ENV="$HOME/qlpc-qaoa-env"
python -m venv "$ENV"
source "$ENV/bin/activate"
python -m pip install --upgrade pip

# Core: qiskit (for parity), the fast backend needs only numpy/scipy; GPU via cupy.
pip install numpy scipy pandas scikit-learn matplotlib
# GPU statevector: cupy matched to the cluster CUDA (12.x shown; use cupy-cuda11x if CUDA 11).
pip install "cupy-cuda12x" || echo "WARN: cupy install failed — study will fall back to CPU (numpy)."
# Optional gate-level parity / cross-check:
pip install "qiskit>=1.0" "qiskit-aer-gpu" || pip install "qiskit>=1.0" qiskit-aer || true

python - <<'PY'
import numpy; print("numpy", numpy.__version__)
try:
    import cupy; cupy.cuda.runtime.getDeviceCount(); print("cupy OK — GPU available")
except Exception as e:
    print("cupy not usable:", type(e).__name__, "(study runs on CPU)")
PY
echo "env ready at $ENV"
