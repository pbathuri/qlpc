#!/bin/bash
#SBATCH --job-name=lcqaoa-study
#SBATCH --partition=gpu                 # BigRed200 GPU partition (A100). Use 'general' for CPU-only.
#SBATCH --account=YOUR_ALLOCATION       # <-- set your IU SLURM allocation/account
#SBATCH --array=0-31                    # 32 shards; instances are split across these tasks
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/study_%A_%a.out
#SBATCH --error=logs/study_%A_%a.err

# Full-scale, no-compromise run: large qubit counts + many instances, sharded across the array.
# Each task computes its slice of (subset x window) instances on one A100 via cupy.
set -euo pipefail
mkdir -p logs

module load python/gpu 2>/dev/null || module load anaconda 2>/dev/null || true
module load cudatoolkit 2>/dev/null || module load cuda 2>/dev/null || true
source "$HOME/qlpc-qaoa-env/bin/activate"

export QLPC_DEVICE=gpu                   # backend.get_xp() -> cupy on the A100
cd "$SLURM_SUBMIT_DIR"                    # submit from the repo root (parent of paper/)

# Full-scale knobs (edit freely). n up to ~26 fits an A100-40GB; ~30 on A100-80GB.
N=22
BUDGET=6
REPS=4
INSTANCES=200                            # per rolling window, per shard-set (sharded below)
Q=2.0

python -m paper.study.run_study \
  --n $N --budget $BUDGET --reps $REPS --instances $INSTANCES --q $Q \
  --shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} \
  --out paper/study/data/study_results.json

echo "shard ${SLURM_ARRAY_TASK_ID} complete"
