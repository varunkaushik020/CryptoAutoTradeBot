"""
Technical indicators: EMA Crossover, RSI, Trendline Breakout, MACD, Bollinger/
Keltner squeeze, ADX, VWAP.
All functions accept a list of OHLCV dicts from Delta Exchange.
"""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from typing import Optional, TypedDict

from bot.lux_indicators import _atr


class IndicatorResult(TypedDict):
    ema_fast: float
    ema_slow: float
    ema_signal: str        # "BULLISH_CROSS" | "BEARISH_CROSS" | "NEUTRAL"
    rsi: float
    rsi_signal: str        # "OVERSOLD" | "OVERBOUGHT" | "NEUTRAL"
    breakout_signal: str   # "BREAKOUT_UP" | "BREAKOUT_DOWN" | "NEUTRAL"
    breakout_level: float | None
    macd: float
    macd_signal_line: float
    macd_hist: float
    macd_signal: str       # "BULLISH_CROSS" | "BEARISH_CROSS" | "BULLISH" | "BEARISH" | "NEUTRAL"
    close: float
    bb_mid: float | None
    bb_upper: float | None
    bb_lower: float | None
    kc_mid: float | None
    kc_upper: float | None
    kc_lower: float | None
    squeeze_on: bool
    squeeze_signal: str     # "SQUEEZE_ON" | "RELEASE_UP" | "RELEASE_DOWN" | "NEUTRAL"
    adx: float | None
    plus_di: float | None
    minus_di: float | None
    vwap: float | None
    vwap_upper1: float | None
    vwap_lower1: float | None
    vwap_upper2: float | None
    vwap_lower2: float | None


def candles_to_df(candles: list[dict]) -> pd.DataFrame:
    """Convert Delta Exchange candle list to a clean DataFrame."""
    df = pd.DataFrame(candles)
    # Delta returns: time, open, high, low, close, volume
    df = df.rename(columns={"time": "timestamp"})
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Flat candles (no gains AND no losses) make RSI undefined -> neutral 50.
    # All-gains (avg_loss==0, avg_gain>0) -> 100. Never leave NaN/inf in output.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return rsi.replace([np.inf, -np.inf], np.nan).fillna(50.0).clip(0, 100)


def calc_trendline_breakout(df: pd.DataFrame, lookback: int = 20) -> tuple[str, float | None]:
    """
    Simple swing-high / swing-low trendline breakout.
    Looks at the last `lookback` candles.
    Returns (signal, level).
    """
    if len(df) < lookback + 2:
        return "NEUTRAL", None

    window = df.tail(lookback + 1)
    resistance = window["high"].iloc[:-1].max()   # highest high in lookback (excluding latest)
    support = window["low"].iloc[:-1].min()       # lowest low in lookback

    latest_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]

    # Breakout UP: price closes above resistance for the first time
    if latest_close > resistance and prev_close <= resistance:
        return "BREAKOUT_UP", float(resistance)

    # Breakout DOWN: price closes below support
    if latest_close < support and prev_close >= support:
        return "BREAKOUT_DOWN", float(support)

    return "NEUTRAL", None


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Classic MACD: EMA(fast) - EMA(slow), signal = EMA(macd), hist = macd - signal.
    Returns (macd_line, signal_line, hist) as pandas Series."""
    macd_line = calc_ema(series, fast) - calc_ema(series, slow)
    signal_line = calc_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calc_bollinger(series: pd.Series, period: int = 20, mult: float = 2.0):
    """Bollinger Bands: SMA mid, +/- mult*stdev bands. Returns (mid, upper, lower)."""
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    return mid, mid + mult * std, mid - mult * std


def calc_keltner(df: pd.DataFrame, candles: list[dict], period: int = 20,
                 atr_mult: float = 1.5, atr_len: int = 10):
    """Keltner Channel: EMA mid, +/- atr_mult*ATR bands. Returns (mid, upper, lower)."""
    mid = calc_ema(df["close"], period)
    atr_series = pd.Series(_atr(candles, atr_len), index=df.index).astype(float)
    return mid, mid + atr_mult * atr_series, mid - atr_mult * atr_series


def calc_squeeze(bb_upper: pd.Series, bb_lower: pd.Series, kc_upper: pd.Series, kc_lower: pd.Series) -> pd.Series:
    """TTM-style squeeze: True when Bollinger Bands sit fully inside the Keltner
    Channel (volatility compressed — a breakout is building)."""
    return (bb_upper < kc_upper) & (bb_lower > kc_lower)


def calc_adx(df: pd.DataFrame, period: int = 14):
    """Wilder ADX/+DI/-DI, smoothed with the same .ewm(com=period-1) convention
    calc_rsi() already uses in this module. Returns (adx, plus_di, minus_di)."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=period - 1, adjust=False).mean()
    safe_atr = atr.replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean() / safe_atr
    minus_di = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / safe_atr
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(com=period - 1, adjust=False).mean()
    return (adx.fillna(0).clip(0, 100), plus_di.fillna(0).clip(0, 100), minus_di.fillna(0).clip(0, 100))


def calc_vwap(candles: list[dict], anchor: str = "session") -> dict:
    """Session-anchored (UTC day) VWAP + stdev bands, using REAL trade volume.

    Delta's MARK-price candles (the bot's default candle source) carry no volume
    (confirmed empirically: `volume` is always None on `MARK:` symbols) — passing
    those in would silently produce a meaningless flat line. Callers must pass
    traded-price candles (get_candles(..., mark=False)); this function also treats
    all-zero/None volume as "no data" and returns Nones rather than guess.
    """
    empty = {"vwap": None, "upper1": None, "lower1": None, "upper2": None, "lower2": None}
    if not candles:
        return empty
    day_keys = [datetime.fromtimestamp(int(c["time"]), tz=timezone.utc).date() for c in candles]
    cur_day = day_keys[-1]
    start = len(candles) - 1
    while start > 0 and day_keys[start - 1] == cur_day:
        start -= 1
    seg = candles[start:]
    seg_vols = [float(c.get("volume") or 0) for c in seg]
    cum_vol = sum(seg_vols)
    if cum_vol <= 0:
        return empty
    typical = [(c["high"] + c["low"] + c["close"]) / 3 for c in seg]
    vwap = sum(tp * v for tp, v in zip(typical, seg_vols)) / cum_vol
    variance = sum(v * (tp - vwap) ** 2 for tp, v in zip(typical, seg_vols)) / cum_vol
    stdev = variance ** 0.5
    return {
        "vwap": round(vwap, 2),
        "upper1": round(vwap + stdev, 2), "lower1": round(vwap - stdev, 2),
        "upper2": round(vwap + 2 * stdev, 2), "lower2": round(vwap - 2 * stdev, 2),
    }


def _safe(v) -> Optional[float]:
    return float(v) if pd.notna(v) else None


def compute_indicators(
    candles: list[dict],
    ema_fast: int = 9,
    ema_slow: int = 21,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_len: int = 9,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_atr_mult: float = 1.5,
    kc_atr_len: int = 10,
    adx_period: int = 14,
    vwap_enabled: bool = True,
    vwap_candles: Optional[list[dict]] = None,
) -> IndicatorResult:
    df = candles_to_df(candles)

    close = df["close"]
    ema_f = calc_ema(close, ema_fast)
    ema_s = calc_ema(close, ema_slow)
    rsi = calc_rsi(close, rsi_period)

    # EMA crossover signal (compare last two candles)
    prev_diff = ema_f.iloc[-2] - ema_s.iloc[-2]
    curr_diff = ema_f.iloc[-1] - ema_s.iloc[-1]
    if prev_diff < 0 and curr_diff > 0:
        ema_signal = "BULLISH_CROSS"
    elif prev_diff > 0 and curr_diff < 0:
        ema_signal = "BEARISH_CROSS"
    else:
        ema_signal = "NEUTRAL"

    # RSI signal
    rsi_val = float(rsi.iloc[-1])
    if rsi_val < rsi_oversold:
        rsi_signal = "OVERSOLD"
    elif rsi_val > rsi_overbought:
        rsi_signal = "OVERBOUGHT"
    else:
        rsi_signal = "NEUTRAL"

    breakout_signal, breakout_level = calc_trendline_breakout(df)

    # MACD (12/26/9): line vs signal, with a fresh-cross flag
    macd_line, macd_sig, macd_hist = calc_macd(close, macd_fast, macd_slow, macd_signal_len)
    macd_v = float(macd_line.iloc[-1])
    macd_sig_v = float(macd_sig.iloc[-1])
    hist_v = float(macd_hist.iloc[-1])
    prev_hist = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else hist_v
    if prev_hist <= 0 and hist_v > 0:
        macd_signal = "BULLISH_CROSS"
    elif prev_hist >= 0 and hist_v < 0:
        macd_signal = "BEARISH_CROSS"
    elif hist_v > 0:
        macd_signal = "BULLISH"
    elif hist_v < 0:
        macd_signal = "BEARISH"
    else:
        macd_signal = "NEUTRAL"

    # Bollinger / Keltner squeeze
    bb_mid, bb_upper, bb_lower = calc_bollinger(close, bb_period, bb_mult)
    kc_mid, kc_upper, kc_lower = calc_keltner(df, candles, kc_period, kc_atr_mult, kc_atr_len)
    squeeze_series = calc_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)
    squeeze_now = bool(squeeze_series.iloc[-1]) if pd.notna(squeeze_series.iloc[-1]) else False
    squeeze_prev = (bool(squeeze_series.iloc[-2])
                    if len(squeeze_series) > 1 and pd.notna(squeeze_series.iloc[-2]) else squeeze_now)
    if squeeze_prev and not squeeze_now:
        squeeze_signal = "RELEASE_UP" if pd.notna(bb_mid.iloc[-1]) and close.iloc[-1] > bb_mid.iloc[-1] else "RELEASE_DOWN"
    elif squeeze_now:
        squeeze_signal = "SQUEEZE_ON"
    else:
        squeeze_signal = "NEUTRAL"

    # ADX trend strength
    adx, plus_di, minus_di = calc_adx(df, adx_period)

    # VWAP: needs REAL volume (mark candles carry none — see calc_vwap docstring).
    vwap_result = calc_vwap(vwap_candles if vwap_candles is not None else candles) if vwap_enabled else {
        "vwap": None, "upper1": None, "lower1": None, "upper2": None, "lower2": None}

    return IndicatorResult(
        ema_fast=float(ema_f.iloc[-1]),
        ema_slow=float(ema_s.iloc[-1]),
        ema_signal=ema_signal,
        rsi=rsi_val,
        rsi_signal=rsi_signal,
        breakout_signal=breakout_signal,
        breakout_level=breakout_level,
        macd=round(macd_v, 4),
        macd_signal_line=round(macd_sig_v, 4),
        macd_hist=round(hist_v, 4),
        macd_signal=macd_signal,
        close=float(close.iloc[-1]),
        bb_mid=_safe(bb_mid.iloc[-1]), bb_upper=_safe(bb_upper.iloc[-1]), bb_lower=_safe(bb_lower.iloc[-1]),
        kc_mid=_safe(kc_mid.iloc[-1]), kc_upper=_safe(kc_upper.iloc[-1]), kc_lower=_safe(kc_lower.iloc[-1]),
        squeeze_on=squeeze_now,
        squeeze_signal=squeeze_signal,
        adx=_safe(adx.iloc[-1]), plus_di=_safe(plus_di.iloc[-1]), minus_di=_safe(minus_di.iloc[-1]),
        vwap=vwap_result["vwap"], vwap_upper1=vwap_result["upper1"], vwap_lower1=vwap_result["lower1"],
        vwap_upper2=vwap_result["upper2"], vwap_lower2=vwap_result["lower2"],
    )
