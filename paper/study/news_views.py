"""Unbiased language input: a local LLM reads REAL news headlines per asset and emits a conviction.

Removes the author from the loop on the language side — the views come from real published headlines, not
from text we wrote to get the result we wanted. Headlines are snapshotted to JSON so the figures stay
reproducible without re-hitting any feed.

Production (historical, point-in-time, no look-ahead): WRDS RavenPack news analytics keyed to the
decision date. Development/validation: the most-recent headlines from a public feed, applied to the most
recent window (so news ~ decision time). The protocol is identical; only the news source differs.
"""
from __future__ import annotations

import json
import os
import urllib.request

from paper import llm_views as LV

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "data", "news_views.json")

SYS = ("You are a disciplined equity analyst. Read the recent headlines for ONE stock and output ONLY a "
       "JSON object {\"conviction\": x} where x in [-1,1]: -1 strongly bearish, 0 neutral/no clear "
       "signal, +1 strongly bullish. Judge only from the headlines; if they are uninformative, output 0.")


def fetch_headlines(ticker: str, k: int = 6) -> list[str]:
    """Recent real headlines for a ticker (yfinance feed). Returns [] on failure (caller treats as neutral)."""
    try:
        import yfinance as yf
        items = yf.Ticker(ticker).news or []
        out = []
        for it in items[:k]:
            t = it.get("title") or (it.get("content", {}) or {}).get("title")
            if t:
                out.append(t.strip())
        return out
    except Exception:  # noqa: BLE001
        return []


def conviction_from_headlines(ticker: str, headlines: list[str],
                              model: str = LV.DEFAULT_MODEL) -> float:
    if not headlines:
        return 0.0
    prompt = (f"{SYS}\n\nStock: {ticker}\nHeadlines:\n- " + "\n- ".join(headlines) + "\n\nJSON only:")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                         "options": {"temperature": 0.0, "seed": 7}}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        raw = json.load(urllib.request.urlopen(req, timeout=120))["response"]
        v = float(json.loads(raw).get("conviction", 0.0))
        return max(-1.0, min(1.0, v))
    except Exception:  # noqa: BLE001
        return 0.0


def build_views(tickers, model: str = LV.DEFAULT_MODEL, refresh: bool = False) -> dict:
    """Real headlines -> per-ticker convictions, cached. Returns {ticker: {conviction, headlines}}."""
    if not refresh and os.path.exists(CACHE):
        return json.load(open(CACHE))
    out = {}
    for t in tickers:
        hl = fetch_headlines(t)
        out[t] = {"conviction": conviction_from_headlines(t, hl, model), "headlines": hl}
    json.dump(out, open(CACHE, "w"), indent=2)
    return out


def conviction_vector(views: dict, tickers) -> dict:
    """{asset_index: conviction} for the QAOA tilt."""
    return {i: float(views.get(t, {}).get("conviction", 0.0)) for i, t in enumerate(tickers)}


if __name__ == "__main__":
    from paper.study.universe import load_prices
    tk = list(load_prices().columns)[:8]
    v = build_views(tk, refresh=True)
    for t in tk:
        print(f"{t}: conv={v[t]['conviction']:+.1f}  ({len(v[t]['headlines'])} headlines)")
