"""Personalization layer — ground the construction in the USER's own cognitive graph (PCE / Cartograph).

This is what makes QLPC *personalized*, not generic: before reasoning, we pull the user's relevant prior
work, conventions, and decisions from their local memory graph, so the LLM explains and constrains the
portfolio in the user's own frameworks (their quant code, risk conventions, prior strategies).

Tries the local `pce` CLI (read-only); degrades to empty context so the pipeline still runs anywhere.
"""
from __future__ import annotations

import json
import shutil
import subprocess

_PCE = r"R:\project-context-engine\.venv\Scripts\pce.exe"


def ground(query: str, limit: int = 4) -> dict:
    """Return {context: [snippets], source}. Queries the user's PCE memory graph if reachable."""
    exe = _PCE if shutil.which(_PCE) or _exists(_PCE) else (shutil.which("pce") or "")
    if exe:
        try:
            p = subprocess.run([exe, "search", query], capture_output=True, text=True, timeout=30)
            snippets = [ln.strip() for ln in (p.stdout or "").splitlines()
                        if ln.strip() and not ln.strip().startswith("-")][:limit]
            if snippets:
                return {"context": snippets, "source": "pce"}
        except Exception:
            pass
    return {"context": [], "source": "none"}


def _exists(path: str) -> bool:
    import os
    return os.path.exists(path)
