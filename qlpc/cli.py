"""QLPC CLI.  python -m qlpc demo   |   python -m qlpc construct "low-drawdown, ~4 names, tail-aware" """
from __future__ import annotations

import argparse
import json
import sys

# Windows consoles default to cp1252 and crash on the box-drawing / math glyphs below — force UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

from .pipeline import construct_portfolio  # noqa: E402


def _report(r: dict) -> None:
    print("\n" + "=" * 70)
    print("  QLPC — Quantum-LLM Personalized portfolio Constructor")
    print("=" * 70)
    print(f"  brief        : {r['brief']}")
    print(f"  constraints  : {r['constraints']}")
    print(f"  data source  : {r['data_source']}    graph grounding: {r['graph']['source']}")
    s = r["selection"]
    print(f"\n  SELECT (quantum): {s['method']}  → {s['tickers']}"
          f"  (optimal vs classical: {s['optimal_vs_classical']})")
    print("  WEIGHTS (convex mean-CVaR):")
    for k, v in r["weights"].items():
        print(f"     {k:6s} {v:6.1%}  " + "█" * int(v * 40))
    rk = r["risk"]
    print(f"\n  RISK (quantum): {rk['method']}   CVaR@{rk['alpha']} = {rk['cvar']:.4f}"
          f"   (classical check {rk['classical_cvar']:.4f}; rel-err {rk.get('rel_err_vs_classical','-')})")
    if "quantum_tail_prob" in rk:
        print(f"     IQAE tail-exceedance P(loss≥VaR) = {rk['quantum_tail_prob']} (target {rk['target_tail_prob']})")
    print(f"  expected return: {r['expected_return']:.2%}")
    print(f"\n  RATIONALE:\n     {r['explanation']}")
    print("=" * 70 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(prog="qlpc")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("demo")
    c = sub.add_parser("construct")
    c.add_argument("brief")
    c.add_argument("--tickers", nargs="*")
    c.add_argument("--live", action="store_true")
    c.add_argument("--json", action="store_true")
    bt = sub.add_parser("backtest")
    bt.add_argument("--tickers", nargs="*")
    bt.add_argument("--live", action="store_true")
    bt.add_argument("--quantum", action="store_true")
    a = ap.parse_args()

    if a.cmd == "construct":
        r = construct_portfolio(a.brief, a.tickers, allow_live=a.live)
        print(json.dumps(r, indent=2)) if a.json else _report(r)
    elif a.cmd == "backtest":
        from .backtest import backtest
        from .pipeline import DEFAULT_UNIVERSE
        r = backtest(a.tickers or DEFAULT_UNIVERSE, allow_live=a.live, quantum=a.quantum)
        print("\n  QLPC dynamic rebalancing backtest")
        for k in ("data_source", "n_rebalances", "annualized_return", "annualized_vol", "sharpe",
                  "max_drawdown", "avg_turnover", "final_equity", "quantum_per_rebalance"):
            print(f"     {k:20s}: {r[k]}")
        print()
    else:  # demo
        _report(construct_portfolio("conservative, low-drawdown, about 4 names, tail-aware",
                                    allow_live=False))


if __name__ == "__main__":
    main()
