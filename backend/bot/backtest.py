"""
Backtest engine — replays historical candles through the SAME strategy votes
and the multi-timeframe (1h trend filter + entry) logic, simulating trades with
an ATR stop and a fixed reward:risk, no pyramiding. Reports per-strategy and
combined performance so you can see which strategies actually have an edge.

Simplifications vs live trading (kept representative, not 1:1):
  - Fixed reward:risk (settings.risk_reward) instead of structure-snapped partials
  - ATR-based stop (the most universal of the live SL methods)
  - One position at a time; exit on SL / TP / opposite signal
"""
import math
import pandas as pd

from bot.indicators import (
    calc_ema, calc_rsi, calc_macd, candles_to_df, calc_bollinger, calc_keltner, calc_squeeze, calc_adx,
)
from bot.lux_indicators import (
    supertrend_ai, trendline_breakout_navigator, fair_value_gaps, inverse_fvg, _atr,
)
from bot import strategies
from bot import smc as smc_mod
from bot import divergence as divergence_mod
from config import settings


def _per_bar_votes(candles: list[dict]) -> dict[str, list[int]]:
    """Return per-bar directional vote (+1/-1/0) for each strategy id."""
    n = len(candles)
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    s = pd.Series(closes)
    ema_f = calc_ema(s, settings.ema_fast).tolist()
    ema_s = calc_ema(s, settings.ema_slow).tolist()
    rsi = calc_rsi(s, settings.rsi_period).tolist()

    st = supertrend_ai(candles)
    os = st["os"] if st else [0] * n
    tn = trendline_breakout_navigator(candles)
    tn_trend = [p["trend"] for p in tn["points"]] if tn else [0] * n

    idx = {int(c["time"]): i for i, c in enumerate(candles)}
    fvg = fair_value_gaps(candles)
    ifvg = inverse_fvg(candles)
    fvg_dir = [0] * n
    ifvg_dir = [0] * n
    for sig in (fvg["signals"] if fvg else []):
        i = idx.get(sig["time"])
        if i is not None:
            fvg_dir[i] = 1 if sig["dir"] == "long" else -1
    for sig in (ifvg["signals"] if ifvg else []):
        i = idx.get(sig["time"])
        if i is not None:
            ifvg_dir[i] = 1 if sig["dir"] == "long" else -1

    # MACD (12/26/9) histogram sign per bar
    macd_hist = (calc_ema(s, 12) - calc_ema(s, 26))
    macd_hist = (macd_hist - calc_ema(macd_hist, 9)).tolist()

    # SMC per-bar structural trend, derived from BOS/CHoCH breaks (one O(n) pass)
    smc_trend = [0] * n
    try:
        brks = sorted(smc_mod._breaks(candles, smc_mod._swings(candles)), key=lambda e: e["i"])
        t, bi = 0, 0
        for i in range(n):
            while bi < len(brks) and brks[bi]["i"] <= i:
                t = 1 if brks[bi]["dir"] == "bullish" else -1
                bi += 1
            smc_trend[i] = t
    except Exception:
        pass

    # Volatility squeeze breakout (BB inside KC, then release) — vectorized over
    # the whole series so it can be replayed bar-by-bar like every other strategy.
    df = candles_to_df(candles)
    bb_mid, bb_upper, bb_lower = calc_bollinger(s, settings.bb_period, settings.bb_mult)
    kc_mid, kc_upper, kc_lower = calc_keltner(df, candles, settings.kc_period, settings.kc_atr_mult, settings.kc_atr_len)
    squeeze_l = calc_squeeze(bb_upper, bb_lower, kc_upper, kc_lower).tolist()
    bb_mid_l = bb_mid.tolist()

    # ADX — used below both for the COMBINED_ADX_GATED A/B and available per-bar
    # for any strategy (VOL_SQUEEZE_BREAKOUT's live conviction scaling is skipped
    # here; backtest votes are directional only, same simplification as every
    # other strategy's strength here).
    adx_l = calc_adx(df, settings.adx_period)[0].tolist()

    # RSI/price divergence: map each confirmed swing signal onto the earliest bar
    # it is actually knowable (sig["i"] + right) — no lookahead.
    div_dir = [0] * n
    try:
        dres = divergence_mod.detect_divergence(candles, settings.divergence_swing_left, settings.divergence_swing_right)
        for sig in (dres or {}).get("signals", []):
            bar = min(n - 1, sig["i"] + settings.divergence_swing_right)
            div_dir[bar] = 1 if sig["dir"] == "bullish" else -1
    except Exception:
        pass

    def fin(x):
        return x if isinstance(x, (int, float)) and math.isfinite(x) else None

    v: dict[str, list[int]] = {sid.value: [0] * n for sid in strategies.StrategyId}
    squeeze_prev = False
    for i in range(n):
        ef, es = fin(ema_f[i]), fin(ema_s[i])
        v["EMA_CROSS"][i] = (1 if ef > es else -1) if (ef is not None and es is not None) else 0
        r = fin(rsi[i])
        v["RSI"][i] = (1 if r < settings.rsi_oversold else -1 if r > settings.rsi_overbought else 0) if r is not None else 0
        # breakout (20-bar)
        if i >= 21:
            res = max(highs[i - 20:i])
            sup = min(lows[i - 20:i])
            v["BREAKOUT"][i] = 1 if (closes[i] > res and closes[i - 1] <= res) else -1 if (closes[i] < sup and closes[i - 1] >= sup) else 0
        v["SUPERTREND_AI"][i] = 1 if os[i] == 1 else -1
        v["TRENDLINE_NAV"][i] = 1 if tn_trend[i] == 1 else -1 if tn_trend[i] == -1 else 0
        v["FVG"][i] = fvg_dir[i]
        v["IFVG"][i] = ifvg_dir[i]
        h = fin(macd_hist[i])
        v["MACD"][i] = (1 if h > 0 else -1 if h < 0 else 0) if h is not None else 0
        v["SMC"][i] = smc_trend[i]
        # squeeze release: vote the breakout direction on the bar it lets go
        sq_now = bool(squeeze_l[i]) if squeeze_l[i] is not None else False
        if squeeze_prev and not sq_now:
            bm = fin(bb_mid_l[i])
            v["VOL_SQUEEZE_BREAKOUT"][i] = 1 if (bm is not None and closes[i] > bm) else -1
        squeeze_prev = sq_now
        v["DIVERGENCE"][i] = div_dir[i]
        # FUNDING_BIAS / ORDERBOOK_IMBALANCE are intentionally NOT backtestable here —
        # no stored history for either exists; they're validated live via shadow mode
        # (see autotune.shadow_status / GET /bot/performance/shadow) instead.
    return v


def _trend_bias_series(candles: list[dict]) -> list[int]:
    """Per-bar 1h-style bias (+1/-1/0) from EMA + SuperTrend + Trendline."""
    n = len(candles)
    closes = [c["close"] for c in candles]
    s = pd.Series(closes)
    ema_f = calc_ema(s, settings.ema_fast).tolist()
    ema_s = calc_ema(s, settings.ema_slow).tolist()
    st = supertrend_ai(candles)
    os = st["os"] if st else [0] * n
    tn = trendline_breakout_navigator(candles)
    tn_trend = [p["trend"] for p in tn["points"]] if tn else [0] * n
    bias = [0] * n
    for i in range(n):
        score = 0
        if math.isfinite(ema_f[i]) and math.isfinite(ema_s[i]):
            score += 1 if ema_f[i] > ema_s[i] else -1
        score += 1 if os[i] == 1 else -1
        score += tn_trend[i]
        bias[i] = 1 if score > 0 else -1 if score < 0 else 0
    return bias


def _simulate(candles, entry_dir, bias, atr, rr, atr_k, use_bias):
    """Walk bars; enter when entry_dir[i] != 0 and (optionally) agrees with 1h bias;
    exit on ATR stop / RR target / opposite signal. Returns trade R-multiples."""
    n = len(candles)
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    trades = []
    pos = 0          # +1 long / -1 short / 0 flat
    entry = sl = tp = 0.0
    for i in range(n):
        if pos != 0:
            hi, lo = highs[i], lows[i]
            exit_p = None
            if pos == 1:
                if lo <= sl: exit_p = sl
                elif hi >= tp: exit_p = tp
            else:
                if hi >= sl: exit_p = sl
                elif lo <= tp: exit_p = tp
            # opposite signal closes at close
            if exit_p is None and entry_dir[i] == -pos:
                exit_p = closes[i]
            if exit_p is not None:
                risk = abs(entry - sl) or 1e-9
                r = (exit_p - entry) / risk * pos
                trades.append(r)
                pos = 0
        if pos == 0 and entry_dir[i] != 0:
            d = entry_dir[i]
            if use_bias and bias is not None and bias[i] != 0 and bias[i] != d:
                continue  # 1h filter blocks counter-trend
            a = atr[i] or 0
            if a <= 0:
                continue
            entry = closes[i]
            sl = entry - a * atr_k if d == 1 else entry + a * atr_k
            tp = entry + a * atr_k * rr if d == 1 else entry - a * atr_k * rr
            pos = d
    return trades


def _metrics(trades: list[float]) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": 0, "avg_r": 0, "total_r": 0, "profit_factor": 0, "max_dd_r": 0, "expectancy": 0}
    wins = [r for r in trades if r > 0]
    losses = [r for r in trades if r <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    # equity curve in R for max drawdown
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in trades:
        eq += r
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    return {
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "avg_r": round(sum(trades) / len(trades), 2),
        "total_r": round(sum(trades), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else (999.0 if gross_win > 0 else 0),
        "max_dd_r": round(max_dd, 2),
        "expectancy": round(sum(trades) / len(trades), 3),
    }


def run_backtest(candles_entry: list[dict], candles_trend: list[dict],
                 rr: float = None, atr_k: float = None, adx_gate_min: float = None) -> dict:
    """Backtest each strategy standalone + the combined engine.

    `adx_gate_min`, when given, also reports a COMBINED_ADX_GATED row that blocks
    entries wherever the 1h ADX was below the threshold — a direct before/after
    comparison to decide config.adx_gate_enabled's default (see config.adx_min_trend).
    """
    rr = rr or settings.risk_reward
    atr_k = atr_k or settings.atr_k
    n = len(candles_entry)
    if n < 60:
        return {"error": "not enough candles"}

    votes = _per_bar_votes(candles_entry)
    atr = _atr(candles_entry, settings.atr_period)

    # map each entry bar to the most recent completed trend (1h) bar's bias / ADX
    trend_bias = _trend_bias_series(candles_trend)
    trend_adx = calc_adx(candles_to_df(candles_trend), settings.adx_period)[0].tolist()
    t_times = [int(c["time"]) for c in candles_trend]
    bias_at = [0] * n
    adx_at = [0.0] * n
    j = 0
    for i, c in enumerate(candles_entry):
        tt = int(c["time"])
        while j + 1 < len(t_times) and t_times[j + 1] <= tt:
            j += 1
        bias_at[i] = trend_bias[j] if t_times and t_times[j] <= tt else 0
        adx_at[i] = trend_adx[j] if t_times and t_times[j] <= tt else 0.0

    results = {}
    # per-strategy (standalone, with 1h filter)
    for sid in strategies.StrategyId:
        trades = _simulate(candles_entry, votes[sid.value], bias_at, atr, rr, atr_k, use_bias=True)
        results[sid.value] = _metrics(trades)

    # combined engine (>= min_signals agreeing, 1h filter)
    enabled = strategies.parse_enabled(settings.strategies)
    combined_dir = [0] * n
    for i in range(n):
        buy = sum(1 for sid in enabled if votes[sid.value][i] == 1)
        sell = sum(1 for sid in enabled if votes[sid.value][i] == -1)
        th = max(1, min(settings.min_signals, len(enabled)))
        combined_dir[i] = 1 if (buy >= th and buy > sell) else -1 if (sell >= th and sell > buy) else 0
    results["COMBINED"] = _metrics(_simulate(candles_entry, combined_dir, bias_at, atr, rr, atr_k, use_bias=True))

    if adx_gate_min is not None:
        gated_dir = [d if adx_at[i] >= adx_gate_min else 0 for i, d in enumerate(combined_dir)]
        results["COMBINED_ADX_GATED"] = _metrics(_simulate(candles_entry, gated_dir, bias_at, atr, rr, atr_k, use_bias=True))

    return {
        "bars": n,
        "rr": rr,
        "atr_k": atr_k,
        "min_signals": settings.min_signals,
        "adx_gate_min": adx_gate_min,
        "results": results,
    }
