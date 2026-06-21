"""Pull CRSP daily total returns for the Dow-30 universe -> the snapshot the study reads.

You run this (it needs your WRDS credentials; an assistant must never enter them). First time, it prompts
for your WRDS password; pass --pgpass to also write ~/.pgpass for passwordless reuse.

    python -m paper.study.wrds_pull --user YOUR_WRDS_USERNAME
    python -m paper.study.wrds_pull --user YOUR_WRDS_USERNAME --start 2022-06-01 --end 2025-06-30

Writes a dividend-adjusted total-return price index to paper/study/data/dow30_prices.csv (the exact
format universe.load_prices expects), so the rest of the study uses gold-standard CRSP data unchanged.
"""
from __future__ import annotations

import argparse

import pandas as pd

from paper.study.universe import DOW30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="your WRDS username")
    ap.add_argument("--start", default="2022-06-01")
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--out", default="paper/study/data/dow30_prices.csv")
    ap.add_argument("--pgpass", action="store_true", help="also create ~/.pgpass for passwordless reuse")
    a = ap.parse_args()

    import wrds
    db = wrds.Connection(wrds_username=a.user)
    if a.pgpass:
        try:
            db.create_pgpass_file()
        except Exception as e:  # noqa: BLE001
            print("(.pgpass not written:", e, ")")
    try:
        tk = ",".join(f"'{t}'" for t in DOW30)
        nm = db.raw_sql(f"select distinct permno, ticker from crsp.stocknames where ticker in ({tk})")
        permnos = ",".join(str(int(p)) for p in nm["permno"].unique())
        dsf = db.raw_sql(
            f"select permno, date, ret from crsp.dsf "
            f"where permno in ({permnos}) and date between '{a.start}' and '{a.end}' and ret is not null",
            date_cols=["date"])
    finally:
        db.close()

    # ticker may map to several permnos (share classes / reuse): keep the most-populated permno per ticker
    pm = {int(r.permno): r.ticker.upper() for r in nm.itertuples()}
    dsf["ticker"] = dsf["permno"].map(pm)
    counts = dsf.groupby(["ticker", "permno"]).size().reset_index(name="n")
    best = counts.sort_values("n").drop_duplicates("ticker", keep="last")
    keep = set(map(tuple, best[["ticker", "permno"]].values))
    dsf = dsf[[(t, p) in keep for t, p in zip(dsf["ticker"], dsf["permno"])]]

    ret = dsf.pivot(index="date", columns="ticker", values="ret").sort_index().dropna()
    px = (1 + ret).cumprod() * 100.0                      # total-return index (includes dividends)
    px.to_csv(a.out)
    print(f"wrote CRSP total-return snapshot: {px.shape[0]} days x {px.shape[1]} names -> {a.out}")
    print("names:", list(px.columns))


if __name__ == "__main__":
    main()
