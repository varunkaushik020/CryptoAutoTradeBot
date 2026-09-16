"""
RSI/price divergence detection at swing points (regular + hidden, both directions).

  Regular bullish:  price Lower Low,  RSI Higher Low   -> reversal up
  Regular bearish:  price Higher High, RSI Lower High  -> reversal down
  Hidden bullish:   price Higher Low, RSI Lower Low    -> uptrend continuation
  Hidden bearish:   price Lower High, RSI Higher High  -> downtrend continuation

Returns the FULL signal history (not just the last bar), the same shape convention
fair_value_gaps()/inverse_fvg() use (a `signals` list + a `latest` pointer), so
bot/backtest.py can replay this strategy bar-by-bar instead of only checking "now".
"""
from __future__ import annotations
from typing import Optional

from bot.indicators import candles_to_df, calc_rsi
from bot.swings import find_swings


def detect_divergence(candles: list[dict], left: int = 2, right: int = 2, rsi_period: int = 14) -> Optional[dict]:
    n = len(candles)
    if n < max(left + right + 5, rsi_period + 5):
        return None
    swings = find_swings(candles, left, right)
    if len(swings) < 2:
        return {"signals": [], "latest": None}

    df = candles_to_df(candles)
    rsi_by_i = calc_rsi(df["close"], rsi_period).tolist()

    signals: list[dict] = []
    last_high = last_low = None  # each: {"i", "price", "rsi"}
    for s in swings:
        i = s["i"]
        if i >= len(rsi_by_i):
            continue
        r = rsi_by_i[i]
        cur = {"i": i, "price": s["price"], "rsi": r}
        if s["kind"] == "high":
            if last_high is not None:
                if cur["price"] > last_high["price"] and cur["rsi"] < last_high["rsi"]:
                    signals.append({"time": s["time"], "i": i, "kind": "regular", "dir": "bearish"})
                elif cur["price"] < last_high["price"] and cur["rsi"] > last_high["rsi"]:
                    signals.append({"time": s["time"], "i": i, "kind": "hidden", "dir": "bearish"})
            last_high = cur
        else:
            if last_low is not None:
                if cur["price"] < last_low["price"] and cur["rsi"] > last_low["rsi"]:
                    signals.append({"time": s["time"], "i": i, "kind": "regular", "dir": "bullish"})
                elif cur["price"] > last_low["price"] and cur["rsi"] < last_low["rsi"]:
                    signals.append({"time": s["time"], "i": i, "kind": "hidden", "dir": "bullish"})
            last_low = cur

    signals.sort(key=lambda x: x["i"])
    latest = None
    if signals:
        latest = dict(signals[-1])
        # "fresh" = the confirming swing landed at (or very near) the newest bar a
        # swing can possibly be confirmed at — an old divergence is not a live signal.
        latest["fresh"] = (n - 1 - latest["i"]) <= (right + 2)
    return {"signals": signals, "latest": latest}
