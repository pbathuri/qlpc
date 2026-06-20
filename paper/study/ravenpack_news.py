"""Point-in-time news via WRDS RavenPack -> per-(date,ticker) LLM conviction. Makes Q3 a VALID backtest.

The local-validation `news_views.py` used recent headlines (look-ahead). This module pulls headlines
TIMESTAMPED BEFORE each rolling-window decision date from RavenPack News Analytics (IU has the full
subscription: ravenpack_full / ravenpack_dj), so the language the optimizer sees at time t is only what
was knowable at t. A local LLM (Ollama) reads those headlines -> conviction, exactly as before; only the
news source (point-in-time, not recent) changes.

You run this (needs your WRDS credentials + Ollama). Two steps:
    python -m paper.study.ravenpack_news --user pbathuri --inspect      # confirm schema (fast)
    python -m paper.study.ravenpack_news --user pbathuri --build        # build the PIT cache (slow)

Writes paper/study/data/pit_news_views.json keyed by decision date; the study then reads it per window.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from paper import llm_views as LV
from paper.study.news_views import conviction_from_headlines
from paper.study.universe import DOW30, load_prices

HERE = os.path.dirname(__file__)
PIT = os.path.join(HERE, "data", "pit_news_views.json")
RAW = os.path.join(HERE, "data", "headlines_raw.json")   # raw RavenPack headlines (gitignored — licensed text)
EQUITY_LIB = "ravenpack_full"            # full edition; rpa_full_equities_<YEAR>
MAP_LIB = "ravenpack_common"


def _cols(db, lib, table):
    try:
        return list(db.describe_table(lib, table)["name"])
    except Exception:  # noqa: BLE001
        return list(db.get_table(lib, table, obs=1).columns)


def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for cand in cands:
        for lc, orig in low.items():
            if cand in lc:
                return orig
    return None


def decision_dates(prices, train=378, test=63):
    """Train-window END dates = the points at which a portfolio is chosen (news must precede these)."""
    d = prices.index; out = []
    for s in range(0, len(d) - train - test, test):
        out.append(d[s + train])
    return out


def entity_map(db, tickers):
    """ticker -> rp_entity_id via the WRDS-curated RavenPack company mapping (schema-adaptive)."""
    tbl = "wrds_rpa_company_mappings"
    cols = _cols(db, MAP_LIB, tbl)
    eid = _pick(cols, "rp_entity_id", "entity_id")
    tk = _pick(cols, "ticker")
    if tk:                                                   # wide form: a ticker column
        tkl = ",".join(f"'{t}'" for t in tickers)
        df = db.raw_sql(f"select distinct {eid}, {tk} from {MAP_LIB}.{tbl} where upper({tk}) in ({tkl})")
        return {r[1].upper(): r[0] for r in df.itertuples(index=False)}, cols
    # long form: data_type/data_value
    dt = _pick(cols, "data_type"); dv = _pick(cols, "data_value")
    tkl = ",".join(f"'{t}'" for t in tickers)
    df = db.raw_sql(f"select distinct {eid}, {dv} from {MAP_LIB}.{tbl} "
                    f"where upper({dt})='TICKER' and upper({dv}) in ({tkl})")
    return {r[1].upper(): r[0] for r in df.itertuples(index=False)}, cols


def headlines_asof(db, entity_id, asof, year, schema, lookback=30, min_rel=90, limit=12):
    tbl = f"rpa_full_equities_{year}"
    ts, hl, eid, rel = schema["ts"], schema["headline"], schema["eid"], schema["relevance"]
    start = (pd.Timestamp(asof) - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")
    end = pd.Timestamp(asof).strftime("%Y-%m-%d")
    relclause = f"and {rel} >= {min_rel}" if rel else ""
    df = db.raw_sql(
        f"select {ts} as ts, {hl} as headline from {EQUITY_LIB}.{tbl} "
        f"where {eid}='{entity_id}' and {ts} >= '{start}' and {ts} < '{end}' {relclause} "
        f"order by {ts} desc limit {limit}")
    return [h for h in df["headline"].tolist() if isinstance(h, str) and h.strip()]


def build(db, tickers, lookback=30, model=LV.DEFAULT_MODEL):
    prices = load_prices()
    dates = decision_dates(prices)
    emap, _ = entity_map(db, tickers)
    print(f"mapped {len(emap)}/{len(tickers)} tickers to RavenPack entities", flush=True)
    eq_cols = _cols(db, EQUITY_LIB, f"rpa_full_equities_{pd.Timestamp(dates[-1]).year}")
    schema = {"ts": _pick(eq_cols, "timestamp_utc", "rpa_date_utc", "timestamp", "date"),
              "headline": _pick(eq_cols, "headline", "news_text", "title"),
              "eid": _pick(eq_cols, "rp_entity_id", "entity_id"),
              "relevance": _pick(eq_cols, "relevance")}
    print("equities schema ->", schema, flush=True)

    cache = {}; raw = {}
    for asof in dates:
        y = pd.Timestamp(asof).year; key = pd.Timestamp(asof).strftime("%Y-%m-%d")
        cache[key] = {}; raw[key] = {}
        for t in tickers:
            eid = emap.get(t.upper())
            hls = headlines_asof(db, eid, asof, y, schema, lookback) if eid else []
            raw[key][t] = hls                              # cache raw text for offline re-extraction
            cache[key][t] = {"conviction": conviction_from_headlines(t, hls, model),
                             "n_headlines": len(hls)}
        nz = sum(1 for t in tickers if abs(cache[key][t]["conviction"]) > 0.01)
        print(f"  {key}: {nz}/{len(tickers)} nonzero convictions", flush=True)
    json.dump({"lookback_days": lookback, "model": model, "decision_dates": list(cache), "views": cache},
              open(PIT, "w"), indent=2)
    json.dump({"lookback_days": lookback, "headlines": raw}, open(RAW, "w"), indent=2)
    print(f"wrote {PIT} and cached raw headlines to {RAW} (re-extract any model offline)", flush=True)


def cache_headlines(db, tickers, lookback=30):
    """Pull raw pre-decision headlines to RAW (WRDS only, NO LLM — fast). Then re-extract any model
    offline with --reextract. Lets us swap LLMs without re-querying WRDS or re-running the slow LLM."""
    prices = load_prices(); dates = decision_dates(prices)
    emap, _ = entity_map(db, tickers)
    yr = pd.Timestamp(dates[-1]).year
    eq_cols = _cols(db, EQUITY_LIB, f"rpa_full_equities_{yr}")
    schema = {"ts": _pick(eq_cols, "timestamp_utc", "rpa_date_utc", "timestamp", "date"),
              "headline": _pick(eq_cols, "headline", "news_text", "title"),
              "eid": _pick(eq_cols, "rp_entity_id", "entity_id"),
              "relevance": _pick(eq_cols, "relevance")}
    raw = {}
    for asof in dates:
        y = pd.Timestamp(asof).year; key = pd.Timestamp(asof).strftime("%Y-%m-%d"); raw[key] = {}
        for t in tickers:
            eid = emap.get(t.upper())
            raw[key][t] = headlines_asof(db, eid, asof, y, schema, lookback) if eid else []
        print(f"  {key}: cached headlines for {len(tickers)} tickers", flush=True)
    json.dump({"lookback_days": lookback, "headlines": raw}, open(RAW, "w"), indent=2)
    print("wrote", RAW, "(now run --reextract --model <m> offline)", flush=True)


def reextract(model, out):
    """Offline: re-run convictions with a DIFFERENT LLM from the cached raw headlines (no WRDS). Used to
    swap in a pre-2023-cutoff model (e.g. Llama 2) so convictions are genuinely out-of-sample, and to
    compare against a model whose training horizon overlaps the evaluation window."""
    raw = json.load(open(RAW))["headlines"]
    ess_src = json.load(open(PIT))["views"] if os.path.exists(PIT) else {}
    views = {}
    for date, tk in raw.items():
        views[date] = {}
        for t, hls in tk.items():
            cv = conviction_from_headlines(t, hls, model=model)
            v = {"conviction": cv, "n_headlines": len(hls)}
            if date in ess_src and t in ess_src[date] and "ess" in ess_src[date][t]:
                v["ess"] = ess_src[date][t]["ess"]         # carry over RavenPack ESS (model-independent)
            views[date][t] = v
        nz = sum(1 for t in tk if abs(views[date][t]["conviction"]) > 0.01)
        print(f"  {date}: {nz}/{len(tk)} nonzero ({model})", flush=True)
    json.dump({"model": model, "decision_dates": list(views), "views": views}, open(out, "w"), indent=2)
    print("wrote", out, flush=True)


def add_ess(db, tickers, lookback=30):
    """Augment the EXISTING cache with RavenPack's own Event Sentiment Score (no LLM, no rebuild).
    Enables the ablation 'does the LLM beat off-the-shelf sentiment?' Adds an 'ess' field per
    (date,ticker), normalized to [-1,1] to match the LLM conviction scale."""
    pit = json.load(open(PIT)); views = pit["views"]
    emap, _ = entity_map(db, tickers)
    yr = pd.Timestamp(list(views)[-1]).year
    eq_cols = _cols(db, EQUITY_LIB, f"rpa_full_equities_{yr}")
    ts = _pick(eq_cols, "timestamp_utc", "rpa_date_utc", "timestamp", "date")
    eid = _pick(eq_cols, "rp_entity_id", "entity_id")
    rel = _pick(eq_cols, "relevance")
    ess = _pick(eq_cols, "event_sentiment_score", "css", "ess")
    print("using ESS column:", ess, flush=True)
    raw = {}
    for date in views:
        asof = pd.Timestamp(date); y = asof.year
        start = (asof - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d"); end = asof.strftime("%Y-%m-%d")
        for t in tickers:
            e = emap.get(t.upper())
            m = None
            if e:
                df = db.raw_sql(f"select avg({ess}) as m from {EQUITY_LIB}.rpa_full_equities_{y} "
                                f"where {eid}='{e}' and {ts} >= '{start}' and {ts} < '{end}' "
                                f"and {rel} >= 90")
                m = df["m"].iloc[0]
            raw[(date, t)] = float(m) if not pd.isna(m) else None
    # detect scale: RavenPack ESS is usually 0-100 (50 neutral); some feeds give [-1,1]
    vals = [v for v in raw.values() if v is not None]
    hi = max(abs(v) for v in vals) if vals else 1.0
    norm = (lambda x: (x - 50.0) / 50.0) if hi > 1.5 else (lambda x: x)
    for (date, t), v in raw.items():
        views[date][t]["ess"] = float(max(-1.0, min(1.0, norm(v)))) if v is not None else 0.0
    json.dump(pit, open(PIT, "w"), indent=2)
    nz = sum(1 for d in views.values() for x in d.values() if abs(x.get("ess", 0)) > 0.01)
    print(f"added ESS to {PIT}: {nz} nonzero (scale {'0-100' if hi>1.5 else '[-1,1]'})", flush=True)


def inspect(db):
    print("MAP table cols:", _cols(db, MAP_LIB, "wrds_rpa_company_mappings"))
    yr = pd.Timestamp(load_prices().index[-1]).year
    print(f"EQUITIES rpa_full_equities_{yr} cols:", _cols(db, EQUITY_LIB, f"rpa_full_equities_{yr}"))
    emap, _ = entity_map(db, DOW30[:5])
    print("sample ticker->entity:", emap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None, help="WRDS username (not needed for --reextract)")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--add-ess", action="store_true", help="add RavenPack ESS to the existing cache (no LLM)")
    ap.add_argument("--model", default=LV.DEFAULT_MODEL, help="Ollama model for conviction extraction")
    ap.add_argument("--cache-only", action="store_true", help="WRDS only: cache raw headlines, no LLM (fast)")
    ap.add_argument("--reextract", action="store_true", help="OFFLINE: re-run convictions on cached headlines")
    ap.add_argument("--out", default=PIT, help="output cache path (for --reextract)")
    ap.add_argument("--lookback", type=int, default=30)
    a = ap.parse_args()
    if a.reextract:                                        # offline path — no WRDS needed
        reextract(a.model, a.out)
        return
    import wrds
    db = wrds.Connection(wrds_username=a.user) if a.user else wrds.Connection()
    try:
        if a.inspect:
            inspect(db)
        if a.cache_only:
            cache_headlines(db, DOW30, a.lookback)
        if a.build:
            build(db, DOW30, a.lookback, a.model)
        if a.add_ess:
            add_ess(db, DOW30, a.lookback)
    finally:
        db.close()


if __name__ == "__main__":
    main()
