"""LLM reasoning layer — local + cloud, with a deterministic fallback so it always runs.

Two jobs:
  1. elicit_constraints: turn a natural-language brief ("low-drawdown, ~5 names, ESG tilt") into the
     optimisation parameters (budget, risk aversion, CVaR alpha, lambda).
  2. explain: a natural-language rationale for the constructed portfolio, GROUNDED in the user's graph
     context so it speaks in their frameworks.

Tries a local Ollama model first (privacy-preserving for holdings), then a cloud key if present; falls
back to a transparent rule-based parser / template. The fallback keeps the prototype fully runnable.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request


def _ollama(prompt: str, model: str = "llama3.1") -> str | None:
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
            headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=20).read())["response"]
    except Exception:
        return None


def elicit_constraints(brief: str) -> dict:
    """NL brief -> {budget, risk_aversion, alpha, lam}. Rule-based (robust); LLM-upgradable."""
    b = brief.lower()
    m = re.search(r"(\d+)\s*(?:names|assets|stocks|holdings)", b)
    budget = int(m.group(1)) if m else (3 if "concentrat" in b else 5)
    if any(w in b for w in ("conservative", "low risk", "low-drawdown", "low drawdown", "defensive")):
        risk_aversion, lam = 8.0, 1.5
    elif any(w in b for w in ("aggressive", "high return", "growth", "risk-on")):
        risk_aversion, lam = 2.0, 0.5
    else:
        risk_aversion, lam = 5.0, 1.0
    alpha = 0.99 if "tail" in b or "crash" in b else 0.95
    return {"budget": budget, "risk_aversion": risk_aversion, "alpha": alpha, "lam": lam}


def explain(result: dict, graph_context: list[str] | None = None) -> str:
    """Natural-language rationale, grounded in the user's graph context when available."""
    sel = result.get("selection", {})
    risk = result.get("risk", {})
    w = result.get("weights", {})
    ctx = graph_context or []
    prompt = (
        "You are a portfolio analyst. In 4-6 sentences, explain this quantum-constructed portfolio to "
        "its owner, grounded in their own prior work below. Be concrete and honest.\n\n"
        f"Selected (QAOA, {selmethod(result)}): {list(w.keys())}\n"
        f"Weights: { {k: round(v,3) for k,v in w.items()} }\n"
        f"Quantum CVaR@{risk.get('alpha')}: {risk.get('cvar')} (method {risk.get('method')}, "
        f"classical {risk.get('classical_cvar')})\n"
        f"User's prior work (graph): {ctx[:3]}\n"
    )
    out = _ollama(prompt) or (os.environ.get("QLPC_CLOUD_LLM") and _cloud(prompt))
    if out:
        return out.strip()
    # deterministic template fallback
    names = ", ".join(w.keys())
    ground = (" Grounded in your prior work: " + " | ".join(c[:90] for c in ctx[:2])) if ctx else ""
    return (f"QLPC selected {len(w)} assets ({names}) via QAOA cardinality optimisation, then solved "
            f"exact convex mean-CVaR weights on that subset. Tail risk was estimated by quantum "
            f"amplitude estimation: CVaR@{risk.get('alpha')} = {risk.get('cvar')} "
            f"(method: {risk.get('method')}; classical check {risk.get('classical_cvar')}).{ground}")


def selmethod(result: dict) -> str:
    return result.get("selection", {}).get("method", "?")


def _cloud(prompt: str) -> str | None:
    return None  # wire a cloud provider here when QLPC_CLOUD_LLM is set; off by default
