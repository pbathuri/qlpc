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

# Scale knobs. NOTE on statistical validity: drawing n of the 30 Dow names — keep n modest so subsets are
# DIVERSE (n=14 of 30 overlaps ~47%; n=22 overlaps ~73% and the instances stop being independent).
# Statistical power comes from MANY diverse instances × decision dates, not from heroic n. To go to large
# n with diversity, switch to a bigger universe (S&P 100): re-pull CRSP + rebuild RavenPack for 100 names.
N=14
BUDGET=5
REPS=4
INSTANCES=400                            # subsets per rolling window, per shard-set (sharded below)
Q=2.0
STRENGTH=0.10                            # realistic +-10% annual view per unit LLM conviction

python -m paper.study.run_study \
  --n $N --budget $BUDGET --reps $REPS --instances $INSTANCES --q $Q --strength $STRENGTH \
  --shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} \
  --out paper/study/data/study_results.json

echo "shard ${SLURM_ARRAY_TASK_ID} complete"
