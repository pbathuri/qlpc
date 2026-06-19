"""Combine SLURM-array shard outputs into one result set + final statistics.
Usage:  python -m paper.study.slurm.aggregate paper/study/data/study_results.shard*.json
"""
import glob
import json
import sys

from paper.study.run_study import summarize


def main(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    rows, cfg = [], None
    for f in sorted(files):
        d = json.load(open(f))
        rows.extend(d["rows"]); cfg = cfg or d["config"]
    merged = {"config": cfg, "rows": rows}
    merged["summary"] = summarize(merged)
    out = "paper/study/data/study_results.json"
    json.dump(merged, open(out, "w"), indent=2)
    print(f"merged {len(files)} shards, {len(rows)} instances -> {out}")
    print(json.dumps(merged["summary"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["paper/study/data/study_results.shard*.json"])
