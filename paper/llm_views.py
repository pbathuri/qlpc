"""LLM view extraction — the real Objective-1 language step.

A local LLM (Ollama) reads natural-language market commentary and emits a signed per-asset CONVICTION
vector v_i in [-1,1]. This is what makes the "language-conditioned" claim literal: the words, not a
hand-set number, produce the tilt that deforms the QAOA cost Hamiltonian.

Deterministic by construction: temperature 0 + JSON-constrained decoding. Reproducible offline only if
Ollama is running; the extracted vector is cached into results.json so figures remain reproducible.
"""
from __future__ import annotations

import json
import urllib.request

OLLAMA = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b-instruct-q4_K_M"

SYSTEM = (
    "You are a disciplined buy-side analyst. Read the market commentary and translate it into a "
    "conviction score for EACH listed ticker. Score scale: -1.0 = strongly bearish, 0.0 = neutral / "
    "no view, +1.0 = strongly bullish. Use the full range and only the evidence in the commentary. "
    "Output ONLY a JSON object mapping every ticker (exact symbol) to its numeric score.")


def extract_views(commentary: str, tickers: list[str], model: str = DEFAULT_MODEL,
                  timeout: int = 180) -> dict:
    """Natural-language commentary -> {ticker: conviction in [-1,1]} via a local LLM. Raises on failure
    (the caller decides whether to fall back), so we never silently fake the language step."""
    prompt = (f"{SYSTEM}\n\nTickers: {', '.join(tickers)}\n\nCommentary:\n\"{commentary}\"\n\n"
              "JSON only:")
    req = urllib.request.Request(
        OLLAMA, data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                                 "format": "json", "options": {"temperature": 0.0, "seed": 7}}).encode(),
        headers={"Content-Type": "application/json"})
    raw = json.load(urllib.request.urlopen(req, timeout=timeout))["response"]
    obj = json.loads(raw)
    out = {}
    for t in tickers:
        v = obj.get(t, obj.get(t.upper(), 0.0))
        try:
            out[t] = float(max(-1.0, min(1.0, float(v))))
        except (TypeError, ValueError):
            out[t] = 0.0
    return out


def views_to_index_map(views: dict, tickers: list[str]) -> dict:
    """{ticker: conviction} -> {asset_index: conviction} for the QAOA tilt (mu -> mu + s*v)."""
    return {i: views.get(t, 0.0) for i, t in enumerate(tickers)}


def available(model: str = DEFAULT_MODEL) -> bool:
    try:
        tags = json.load(urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5))
        return any(m["name"] == model for m in tags.get("models", []))
    except Exception:  # noqa: BLE001
        return False


# Commentaries used in the paper (fixed text -> the LLM does the rest). Graded conviction on the same
# thesis lets us show the optimum migrate as language strength rises.
COMMENTARIES = {
    "neutral": "No strong directional views across the names; markets look broadly balanced.",
    "moderate": ("Gold is attractive as a macro hedge and energy looks firm on supply discipline; "
                 "mega-cap tech is fine but fully valued here."),
    "strong": ("I am decisively rotating out of expensive mega-cap tech into real assets: gold is my "
               "highest-conviction long as a macro hedge, and energy benefits from tight supply. "
               "Apple and Microsoft look stretched and I would underweight them."),
}


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "NVDA", "XOM", "JPM", "GLD"]
    print("model available:", available())
    for k, c in COMMENTARIES.items():
        print(k, "->", extract_views(c, tickers))
