"""
Shared fractal swing-high/low detection.

Extracted from smc.py (the richest of several near-duplicate swing finders that had
grown independently in smc.py, scheduler.py and ai_brain.py) so new code — like
divergence detection — has one canonical implementation to build on instead of a
fourth reimplementation. Behavior is unchanged from the original smc._swings.
"""
from __future__ import annotations


def find_swings(candles: list[dict], left: int = 2, right: int = 2) -> list[dict]:
    """Fractal swing highs/lows (confirmed `right` bars later). Sorted by index."""
    n = len(candles)
    out: list[dict] = []
    for i in range(left, n - right):
        hi, lo = candles[i]["high"], candles[i]["low"]
        if all(hi > candles[j]["high"] for j in range(i - left, i)) and \
           all(hi >= candles[j]["high"] for j in range(i + 1, i + right + 1)):
            out.append({"i": i, "time": int(candles[i]["time"]), "price": round(hi, 2), "kind": "high"})
        if all(lo < candles[j]["low"] for j in range(i - left, i)) and \
           all(lo <= candles[j]["low"] for j in range(i + 1, i + right + 1)):
            out.append({"i": i, "time": int(candles[i]["time"]), "price": round(lo, 2), "kind": "low"})
    out.sort(key=lambda s: s["i"])
    return out
