"""Check whether your WRDS subscription includes point-in-time news (RavenPack) for a valid Q3 backtest,
and discover the exact library/table names. You run this (needs your WRDS credentials).

    python -m paper.study.wrds_explore --user YOUR_WRDS_USERNAME

Paste the output back so the point-in-time `news_views` variant can be wired to the right tables.
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    a = ap.parse_args()

    import wrds
    db = wrds.Connection(wrds_username=a.user)
    try:
        libs = db.list_libraries()
        news_libs = [l for l in libs if any(k in l.lower() for k in ("raven", "news", "rpna", "tr_news"))]
        print("candidate news/sentiment libraries:", news_libs or "(none found)")
        for lib in news_libs:
            try:
                print(f"\n[{lib}] tables:", db.list_tables(library=lib)[:30])
            except Exception as e:  # noqa: BLE001
                print(f"[{lib}] could not list tables:", e)
        # also surface CRSP libs so the data pull is confirmed available
        print("\nCRSP libraries:", [l for l in libs if l.lower().startswith("crsp")][:10])
    finally:
        db.close()


if __name__ == "__main__":
    main()
