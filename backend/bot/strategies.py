"""
Pluggable strategy/indicator engine.

Every signal source is a registered function keyed by a `StrategyId` enum.
The bot evaluates ALL enabled strategies, each casting a BUY / SELL / NEUTRAL
vote, then aggregates them (needs `min_signals` agreeing votes to trade).

Adding a new indicator/strategy is a 2-line change:
    1. add a value to `StrategyId`
    2. write a function decorated with `@register(StrategyId.NEW_ONE)`
…and (optionally) add its id to the `STRATEGIES` setting to enable it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NEUTRAL = "NEUTRAL"


class StrategyId(str, Enum):
    EMA_CROSS = "EMA_CROSS"
    RSI = "RSI"
    BREAKOUT = "BREAKOUT"
    SUPERTREND_AI = "SUPERTREND_AI"
    TRENDLINE_NAV = "TRENDLINE_NAV"
    FVG = "FVG"
    IFVG = "IFVG"
    SMC = "SMC"
    MACD = "MACD"
    VOL_SQUEEZE_BREAKOUT = "VOL_SQUEEZE_BREAKOUT"
    DIVERGENCE = "DIVERGENCE"
    # Crypto-native, shadow-mode-validated (see config.shadow_strategies): these cast
    # real votes only once promoted into the live `strategies` CSV.
    FUNDING_BIAS = "FUNDING_BIAS"
    ORDERBOOK_IMBALANCE = "ORDERBOOK_IMBALANCE"


@dataclass
class Signal:
    action: Action
    reason: str = ""
    strength: float = 1.0


@dataclass
class StrategyContext:
    """Everything a strategy might need, precomputed once per tick."""
    candles: list
    ind: dict                      # compute_indicators() output (EMA/RSI/breakout)
    supertrend: Optional[dict] = None
    trendline: Optional[dict] = None
    fvg: Optional[dict] = None
    ifvg: Optional[dict] = None
    extra: dict = field(default_factory=dict)


StrategyFn = Callable[[StrategyContext], Signal]
REGISTRY: dict[StrategyId, StrategyFn] = {}
LABELS: dict[StrategyId, str] = {
    StrategyId.EMA_CROSS: "EMA 9/21 Crossover",
    StrategyId.RSI: "RSI Reversal",
    StrategyId.BREAKOUT: "Trendline Breakout (basic)",
    StrategyId.SUPERTREND_AI: "SuperTrend AI (LuxAlgo)",
    StrategyId.TRENDLINE_NAV: "Trendline Breakout Navigator (LuxAlgo)",
    StrategyId.FVG: "Fair Value Gap (LuxAlgo)",
    StrategyId.IFVG: "Inversion Fair Value Gap (LuxAlgo)",
    StrategyId.SMC: "Smart Money Concepts (structure/liquidity/OB)",
    StrategyId.MACD: "MACD (12/26/9)",
    StrategyId.VOL_SQUEEZE_BREAKOUT: "Volatility Squeeze Breakout (BB/KC)",
    StrategyId.DIVERGENCE: "RSI/Price Divergence",
    StrategyId.FUNDING_BIAS: "Funding Rate Bias (contrarian, shadow)",
    StrategyId.ORDERBOOK_IMBALANCE: "Order-Book Imbalance (shadow)",
}


def register(sid: StrategyId):
    def deco(fn: StrategyFn) -> StrategyFn:
        REGISTRY[sid] = fn
        return fn
    return deco


# --------------------------------------------------------------------------- #
#  Strategy implementations
# --------------------------------------------------------------------------- #
@register(StrategyId.EMA_CROSS)
def _ema_cross(ctx: StrategyContext) -> Signal:
    # Continuous trend vote: EMA9 above EMA21 = bullish, below = bearish.
    fast = ctx.ind.get("ema_fast")
    slow = ctx.ind.get("ema_slow")
    if fast is None or slow is None:
        return Signal(Action.NEUTRAL)
    crossed = ctx.ind.get("ema_signal")  # extra conviction if it just crossed
    if fast > slow:
        return Signal(Action.BUY, "EMA9 above EMA21 (uptrend)" + (" — fresh cross" if crossed == "BULLISH_CROSS" else ""),
                      1.3 if crossed == "BULLISH_CROSS" else 1.0)
    if fast < slow:
        return Signal(Action.SELL, "EMA9 below EMA21 (downtrend)" + (" — fresh cross" if crossed == "BEARISH_CROSS" else ""),
                      1.3 if crossed == "BEARISH_CROSS" else 1.0)
    return Signal(Action.NEUTRAL)


@register(StrategyId.RSI)
def _rsi(ctx: StrategyContext) -> Signal:
    s = ctx.ind.get("rsi_signal")
    rsi = ctx.ind.get("rsi", 0)
    if s == "OVERSOLD":
        return Signal(Action.BUY, f"RSI {rsi:.1f} oversold — bounce expected")
    if s == "OVERBOUGHT":
        return Signal(Action.SELL, f"RSI {rsi:.1f} overbought — pullback expected")
    return Signal(Action.NEUTRAL)


@register(StrategyId.BREAKOUT)
def _breakout(ctx: StrategyContext) -> Signal:
    s = ctx.ind.get("breakout_signal")
    lvl = ctx.ind.get("breakout_level")
    if s == "BREAKOUT_UP":
        return Signal(Action.BUY, f"price broke above resistance {lvl:.0f}")
    if s == "BREAKOUT_DOWN":
        return Signal(Action.SELL, f"price broke below support {lvl:.0f}")
    return Signal(Action.NEUTRAL)


@register(StrategyId.SUPERTREND_AI)
def _supertrend_ai(ctx: StrategyContext) -> Signal:
    st = ctx.supertrend
    if not st:
        return Signal(Action.NEUTRAL)
    latest = st.get("latest", {})
    strength = latest.get("strength", 0)
    fresh = " — fresh flip" if latest.get("flip") else ""
    # Continuous trend vote from the SuperTrend direction (os).
    if latest.get("dir") == "long":
        return Signal(Action.BUY, f"SuperTrend AI bullish (strength {strength}/10){fresh}", 1.0 + strength / 10)
    if latest.get("dir") == "short":
        return Signal(Action.SELL, f"SuperTrend AI bearish (strength {strength}/10){fresh}", 1.0 + strength / 10)
    return Signal(Action.NEUTRAL)


@register(StrategyId.TRENDLINE_NAV)
def _trendline_nav(ctx: StrategyContext) -> Signal:
    tn = ctx.trendline
    if not tn:
        return Signal(Action.NEUTRAL)
    latest = tn.get("latest", {})
    fresh = " — fresh break" if latest.get("flip") else ""
    # Continuous trend vote from the Navigator's current trend.
    if latest.get("trend") == 1:
        return Signal(Action.BUY, f"Trendline Navigator bullish{fresh}")
    if latest.get("trend") == -1:
        return Signal(Action.SELL, f"Trendline Navigator bearish{fresh}")
    return Signal(Action.NEUTRAL)


@register(StrategyId.FVG)
def _fvg(ctx: StrategyContext) -> Signal:
    fvg = ctx.fvg
    if not fvg:
        return Signal(Action.NEUTRAL)
    latest = fvg.get("latest", {})
    if latest.get("new"):
        if latest.get("dir") == "long":
            return Signal(Action.BUY, "Bullish Fair Value Gap formed (imbalance up)")
        if latest.get("dir") == "short":
            return Signal(Action.SELL, "Bearish Fair Value Gap formed (imbalance down)")
    return Signal(Action.NEUTRAL)


@register(StrategyId.IFVG)
def _ifvg(ctx: StrategyContext) -> Signal:
    ifvg = ctx.ifvg
    if not ifvg:
        return Signal(Action.NEUTRAL)
    latest = ifvg.get("latest", {})
    if latest.get("new"):
        if latest.get("dir") == "long":
            return Signal(Action.BUY, "Inversion FVG reclaimed as support (bullish)")
        if latest.get("dir") == "short":
            return Signal(Action.SELL, "Inversion FVG rejected as resistance (bearish)")
    return Signal(Action.NEUTRAL)


@register(StrategyId.MACD)
def _macd(ctx: StrategyContext) -> Signal:
    """MACD (12/26/9): histogram sign gives the vote; a fresh cross adds conviction,
    and MACD above/below the zero line confirms the trend side."""
    sig = ctx.ind.get("macd_signal")
    if not sig or sig == "NEUTRAL":
        return Signal(Action.NEUTRAL)
    macd = ctx.ind.get("macd", 0.0)
    if sig in ("BULLISH_CROSS", "BULLISH"):
        fresh = sig == "BULLISH_CROSS"
        strength = (1.3 if fresh else 1.0) + (0.2 if macd > 0 else 0.0)
        return Signal(Action.BUY, f"MACD {'bullish cross' if fresh else 'histogram positive'}"
                      + (" above zero" if macd > 0 else ""), strength)
    if sig in ("BEARISH_CROSS", "BEARISH"):
        fresh = sig == "BEARISH_CROSS"
        strength = (1.3 if fresh else 1.0) + (0.2 if macd < 0 else 0.0)
        return Signal(Action.SELL, f"MACD {'bearish cross' if fresh else 'histogram negative'}"
                      + (" below zero" if macd < 0 else ""), strength)
    return Signal(Action.NEUTRAL)


@register(StrategyId.SMC)
def _smc(ctx: StrategyContext) -> Signal:
    """Smart Money Concepts confluence vote: score structure + sweep + premium/discount
    + unmitigated order block, aligned with the higher-timeframe SMC trend."""
    s = ctx.extra.get("smc")
    if not s:
        return Signal(Action.NEUTRAL)
    st = ctx.extra.get("smc_trend") or {}
    htf = st.get("trend")  # bullish / bearish / ranging

    sb = s.get("structure_break") or {}
    sweep = s.get("recent_sweep") or {}
    pd = s.get("premium_discount") or {}
    obs = s.get("order_blocks") or {}
    zone = pd.get("zone")

    bull = bear = 0
    reasons = []
    if s.get("trend") == "bullish": bull += 1
    if s.get("trend") == "bearish": bear += 1
    if sb.get("dir") == "bullish": bull += 1; reasons.append(f"{sb.get('type')}↑")
    if sb.get("dir") == "bearish": bear += 1; reasons.append(f"{sb.get('type')}↓")
    if sweep.get("dir") == "bullish": bull += 1; reasons.append("sell-side sweep")
    if sweep.get("dir") == "bearish": bear += 1; reasons.append("buy-side sweep")
    if zone == "discount": bull += 1
    if zone == "premium": bear += 1
    if any(not o.get("mitigated") for o in obs.get("bullish", [])): bull += 1; reasons.append("bull OB")
    if any(not o.get("mitigated") for o in obs.get("bearish", [])): bear += 1; reasons.append("bear OB")

    # require higher-timeframe alignment for a vote (core SMC rule)
    if bull >= 2 and bull > bear and htf != "bearish":
        return Signal(Action.BUY, "SMC bullish: " + ", ".join(reasons), min(1.6, 1.0 + 0.15 * bull))
    if bear >= 2 and bear > bull and htf != "bullish":
        return Signal(Action.SELL, "SMC bearish: " + ", ".join(reasons), min(1.6, 1.0 + 0.15 * bear))
    return Signal(Action.NEUTRAL)


@register(StrategyId.VOL_SQUEEZE_BREAKOUT)
def _vol_squeeze_breakout(ctx: StrategyContext) -> Signal:
    """TTM-style squeeze release: votes the breakout direction the instant a BB-
    inside-KC volatility squeeze lets go, scaled up by ADX (a stronger trend behind
    the release earns more conviction)."""
    sig = ctx.ind.get("squeeze_signal")
    if sig not in ("RELEASE_UP", "RELEASE_DOWN"):
        return Signal(Action.NEUTRAL)
    adx = ctx.ind.get("adx") or 0
    strength = 1.0 + min(adx, 40) / 40 * 0.5
    if sig == "RELEASE_UP":
        return Signal(Action.BUY, f"Volatility squeeze released up (ADX {adx:.0f})", strength)
    return Signal(Action.SELL, f"Volatility squeeze released down (ADX {adx:.0f})", strength)


@register(StrategyId.DIVERGENCE)
def _divergence(ctx: StrategyContext) -> Signal:
    """RSI/price divergence at swing points. Regular divergence votes the reversal
    direction; hidden divergence votes trend continuation. Only a FRESH divergence
    (confirmed at/near the latest bar) casts a vote — a stale one is not a signal."""
    d = ctx.extra.get("divergence")
    latest = (d or {}).get("latest")
    if not latest or not latest.get("fresh"):
        return Signal(Action.NEUTRAL)
    label = f"{latest.get('kind')} {latest.get('dir')} RSI divergence"
    if latest.get("dir") == "bullish":
        return Signal(Action.BUY, label)
    if latest.get("dir") == "bearish":
        return Signal(Action.SELL, label)
    return Signal(Action.NEUTRAL)


@register(StrategyId.FUNDING_BIAS)
def _funding_bias(ctx: StrategyContext) -> Signal:
    """Contrarian fade of extreme perpetual funding: crowded longs paying heavily
    tend to unwind, and vice versa. SHADOW-mode by default (config.shadow_strategies)
    — tracked on a side ledger until it proves positive expectancy live."""
    f = ctx.extra.get("funding")
    extreme = (f or {}).get("extreme")
    rate = (f or {}).get("funding_rate")
    if extreme == "high":
        return Signal(Action.SELL, f"Funding extremely high ({rate}) — crowded longs, fade")
    if extreme == "low":
        return Signal(Action.BUY, f"Funding extremely low ({rate}) — crowded shorts, fade")
    return Signal(Action.NEUTRAL)


@register(StrategyId.ORDERBOOK_IMBALANCE)
def _orderbook_imbalance(ctx: StrategyContext) -> Signal:
    """Short-horizon L2 book bid/ask volume skew. SHADOW-mode by default (config.
    shadow_strategies) — a seconds-scale signal cast at low conviction given the
    horizon mismatch with the 15m decision cadence it would otherwise vote into."""
    ob = ctx.extra.get("orderbook_imbalance")
    sig = (ob or {}).get("signal")
    imb = (ob or {}).get("imb")
    if sig == "BUY":
        return Signal(Action.BUY, f"Order book bid-heavy (imbalance {imb:+.2f})", 0.5)
    if sig == "SELL":
        return Signal(Action.SELL, f"Order book ask-heavy (imbalance {imb:+.2f})", 0.5)
    return Signal(Action.NEUTRAL)


# --------------------------------------------------------------------------- #
#  Engine
# --------------------------------------------------------------------------- #
def parse_enabled(csv: str) -> list[StrategyId]:
    out = []
    for tok in (csv or "").split(","):
        tok = tok.strip().upper()
        if not tok:
            continue
        try:
            out.append(StrategyId(tok))
        except ValueError:
            pass
    return out or [StrategyId.EMA_CROSS, StrategyId.RSI, StrategyId.BREAKOUT]


def evaluate(ctx: StrategyContext, enabled: list[StrategyId], min_signals: int,
             weights: dict | None = None, shadow: list[StrategyId] | None = None) -> dict:
    """
    Run every enabled strategy and aggregate votes into a decision.
    `weights` (auto-tune) scales each strategy's vote: still need `min_signals`
    distinct strategies agreeing, but the WEIGHTED score decides which side wins.

    `shadow` strategies (e.g. FUNDING_BIAS, ORDERBOOK_IMBALANCE — see
    config.shadow_strategies) are evaluated too, but returned separately as
    `shadow_votes` and NEVER folded into buy/sell/threshold — they build a track
    record (via autotune.record_shadow_outcome) without ever influencing a real
    trade until promoted into `enabled`.
    """
    weights = weights or {}
    shadow_votes: dict[str, str] = {}
    for sid in (shadow or []):
        if sid in enabled:
            continue  # already a real, voting strategy — no separate shadow ledger needed
        fn = REGISTRY.get(sid)
        if not fn:
            continue
        try:
            shadow_votes[sid.value] = fn(ctx).action.value
        except Exception as e:
            shadow_votes[sid.value] = Signal(Action.NEUTRAL, f"error: {e}").action.value
    votes: dict[str, str] = {}
    reasons: list[str] = []
    buy = sell = 0          # counts (for the min_signals gate)
    buy_w = sell_w = 0.0    # weighted scores (for the decision + conviction)

    for sid in enabled:
        fn = REGISTRY.get(sid)
        if not fn:
            continue
        try:
            sig = fn(ctx)
        except Exception as e:
            sig = Signal(Action.NEUTRAL, f"error: {e}")
        votes[sid.value] = sig.action.value
        wt = weights.get(sid.value, 1.0)
        if sig.action == Action.BUY:
            buy += 1
            buy_w += wt
            reasons.append(f"[{sid.value}×{wt:g}] {sig.reason}")
        elif sig.action == Action.SELL:
            sell += 1
            sell_w += wt
            reasons.append(f"[{sid.value}×{wt:g}] {sig.reason}")

    threshold = max(1, min(min_signals, len(enabled)))
    if buy >= threshold and buy_w > sell_w:
        action = Action.BUY
    elif sell >= threshold and sell_w > buy_w:
        action = Action.SELL
    else:
        action = Action.HOLD

    reason = " | ".join(reasons) if reasons else \
        f"No strong signal (buy={buy}, sell={sell}, need {threshold})"
    return {
        "action": action.value,
        "reason": reason,
        "votes": votes,
        "shadow_votes": shadow_votes,
        "buy_score": buy,
        "sell_score": sell,
        "buy_w": round(buy_w, 2),
        "sell_w": round(sell_w, 2),
        "threshold": threshold,
    }


def available() -> list[dict]:
    """For UI: list every registered strategy with a human label."""
    return [{"id": sid.value, "label": LABELS.get(sid, sid.value)} for sid in StrategyId if sid in REGISTRY]
