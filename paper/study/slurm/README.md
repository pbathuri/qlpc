# Running the study on IU BigRed200 (no-compromise, full scale)

The study is embarrassingly parallel over `(asset-subset × rolling-window)` instances, so it maps cleanly
onto a SLURM job array, one A100 GPU per task via `cupy`. This is where the large qubit counts and the
hundreds of instances run; locally we only validate the pipeline at moderate `n`.

## 0. Why HPC is actually required here
Exact QAOA statevector simulation is `O(2^n)` in memory and per-evaluation time. On a workstation CPU a
single `p=4` evaluation is ~1.3 s at `n=20` and ~6 s at `n=22`; an instance needs hundreds of
evaluations, and the study needs hundreds of instances. An A100 via `cupy` is ~10–50× faster per
evaluation, and the array fans the instances across many GPUs. This is a genuine compute requirement, not
an optimization we skipped — small-`n`/few-instance shortcuts are exactly the bias we are removing.

## 1. Copy the repo and set up the env (once)
```bash
ssh <user>@bigred200.uits.iu.edu
# put the repo under e.g. /N/slate/<user>/qlpc   (scratch — fast, large)
cd /N/slate/<user>/qlpc/paper/study/slurm
bash setup_env.sh                 # creates ~/qlpc-qaoa-env with numpy/scipy/cupy(+qiskit)
```

## 2. Provide the data
- **Gold standard (recommended):** pull CRSP daily returns once with your WRDS credentials
  (`paper.data_layer.wrds_crsp_returns`) — you run this; an assistant must never enter WRDS credentials.
  For news, WRDS **RavenPack** gives point-in-time headlines (no look-ahead) keyed to the decision date.
- **Offline-reproducible:** the committed `paper/study/data/dow30_prices.csv` snapshot + cached
  `news_views.json` already ship in the repo, so the array runs with zero external calls.

## 3. Submit the array
```bash
cd /N/slate/<user>/qlpc            # submit from the repo root (parent of paper/)
# edit run_array.slurm: set --account, and N / BUDGET / REPS / INSTANCES for the scale you want
sbatch paper/study/slurm/run_array.slurm
squeue -u $USER
```
Each task writes `paper/study/data/study_results.shard<k>.json`.

## 4. Aggregate + final statistics
```bash
source ~/qlpc-qaoa-env/bin/activate
python -m paper.study.slurm.aggregate "paper/study/data/study_results.shard*.json"
# -> paper/study/data/study_results.json with Q1/Q2/Q3 summary (bootstrap CI + Wilcoxon)
```

## 5. Figures
```bash
python -m paper.study.figures        # distribution plots from study_results.json (CPU, seconds)
```

## Scale guidance (per A100)
| GPU | approx max n (statevector) | suggested study n |
|-----|----------------------------|-------------------|
| A100 40 GB | ~31 qubits (complex64) | 22–26 |
| A100 80 GB | ~32–33 qubits | 26–30 |

Keep `n` modest and `INSTANCES` large: statistical power over many honest instances beats one heroic
large-`n` run. Report the full distribution; never select the favorable instance.
