"""
Delta Exchange API client (testnet by default).
Docs: https://docs.delta.exchange/
"""
import asyncio
import hashlib
import hmac
import json
import logging
import math
import time
from typing import Optional
import httpx
from config import settings

logger = logging.getLogger("bot.delta_client")

# Delta only accepts these resolution strings for history candles.
_ALLOWED_RESOLUTIONS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"}


_MINUTE_RESOLUTION = {
    1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m",
    60: "1h", 120: "2h", 240: "4h", 360: "6h", 720: "12h",
    1440: "1d", 10080: "1w",
}


def vwap_fill(levels: list[dict], size: float) -> tuple[float, float]:
    """Walk book `levels` to fill `size` contracts.

    Returns (volume-weighted fill price, size actually fillable). A market order eats
    the book level by level, so the realistic exit price is this VWAP — not the top
    of book, and definitely not the mark price.
    """
    need, cost, got = float(size), 0.0, 0.0
    for lv in levels or []:
        try:
            px, sz = float(lv["price"]), float(lv["size"])
        except (KeyError, TypeError, ValueError):
            continue
        take = min(need, sz)
        if take <= 0:
            break
        cost += px * take
        got += take
        need -= take
        if need <= 1e-9:
            break
    return (cost / got if got else 0.0), got


def minutes_to_resolution(minutes: int) -> str:
    """Convert an interval in minutes to a Delta resolution string (e.g. 5 -> '5m', 240 -> '4h', 1440 -> '1d')."""
    if minutes in _MINUTE_RESOLUTION:
        return _MINUTE_RESOLUTION[minutes]
    if minutes % 60 == 0 and minutes >= 60:
        candidate = f"{minutes // 60}h"
    else:
        candidate = f"{minutes}m"
    return candidate if candidate in _ALLOWED_RESOLUTIONS else "5m"


class DeltaClient:
    def __init__(self):
        self.base_url = settings.delta_base_url.rstrip("/")
        self.api_key = settings.delta_api_key
        self.api_secret = settings.delta_api_secret
        self._product_cache: dict[str, int] = {}
        self._cv_cache: dict[str, float] = {}
        self._rcache: dict[str, tuple[float, object]] = {}   # short-TTL read cache
        self._rlocks: dict[str, asyncio.Lock] = {}           # in-flight dedup per key

    async def _cached(self, key: str, ttl: float, factory):
        """Return a cached read (shared across all callers) or fetch once."""
        ent = self._rcache.get(key)
        if ent and time.time() - ent[0] < ttl:
            return ent[1]
        lock = self._rlocks.setdefault(key, asyncio.Lock())
        async with lock:
            ent = self._rcache.get(key)
            if ent and time.time() - ent[0] < ttl:
                return ent[1]
            val = await factory()
            self._rcache[key] = (time.time(), val)
            return val

    def invalidate_cache(self):
        """Drop cached reads (call right after placing/cancelling orders)."""
        self._rcache.clear()

    def _sign(self, method: str, path: str, query: str = "", payload: str = "") -> dict:
        # Delta signs: method + timestamp + requestPath + query_string + body
        timestamp = str(int(time.time()))
        message = method + timestamp + path + query + payload
        signature = hmac.new(
            self.api_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        return {
            "api-key": self.api_key,
            "timestamp": timestamp,
            "signature": signature,
            "Content-Type": "application/json",
            "User-Agent": "forexbot/1.0",
        }

    async def get_candles(self, symbol: str, resolution: int = 5, limit: int = 100,
                          mark: Optional[bool] = None) -> list[dict]:
        """
        Fetch OHLCV candles.
        resolution: candle size in minutes (5 = 5m candles).
        mark: force mark-price (True) or traded-price (False) candles. Default (None)
              follows settings.use_mark_candles. Traded candles carry volume; mark
              candles are smoother but have no volume.
        """
        res_str = minutes_to_resolution(resolution)
        end_time = int(time.time())
        # Widen the window 1.5x so gaps in low-liquidity testnet data still yield `limit` candles.
        start_time = end_time - int(resolution * 60 * limit * 1.5)
        # Prefer the index-derived MARK price: smooth + accurate, unlike the thin
        # last-traded feed which prints fake wicks on the demo/testnet.
        use_mark = settings.use_mark_candles if mark is None else mark
        candle_symbol = (f"MARK:{symbol}"
                         if use_mark and not symbol.startswith("MARK:")
                         else symbol)
        url = f"{self.base_url}/v2/history/candles"
        params = {
            "resolution": res_str,
            "symbol": candle_symbol,
            "start": start_time,
            "end": end_time,
        }
        # Delta's candle endpoint intermittently returns a spurious 400 under load,
        # so retry a few times with backoff before giving up.
        data = None
        last_err: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=15) as client:
            for attempt in range(4):
                try:
                    resp = await client.get(url, params=params, headers={"User-Agent": "forexbot/1.0"})
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except httpx.HTTPStatusError as e:
                    last_err = e
                    logger.warning("Candles request failed (attempt %d/4): %s", attempt + 1, e)
                    await asyncio.sleep(0.5 * (attempt + 1))
            if data is None:
                raise last_err  # type: ignore[misc]
            raw = data.get("result", []) or []
            normalized = []
            for c in raw:
                try:
                    if isinstance(c, dict):
                        t = c.get("time")
                        o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
                        vol = c.get("volume", 0)
                    elif isinstance(c, list) and len(c) >= 5:
                        t, o, h, l, cl = c[0], c[1], c[2], c[3], c[4]
                        vol = c[5] if len(c) > 5 else 0
                    else:
                        continue
                    if t is None or o is None or h is None or l is None or cl is None:
                        continue
                    candle = {
                        "time": int(t), "open": float(o), "high": float(h),
                        "low": float(l), "close": float(cl), "volume": float(vol or 0),
                    }
                    # Skip any candle that didn't produce finite numbers.
                    if not all(math.isfinite(candle[k]) for k in ("open", "high", "low", "close")):
                        continue
                    normalized.append(candle)
                except (TypeError, ValueError, KeyError, IndexError):
                    continue
            normalized.sort(key=lambda x: x["time"])
            return normalized[-limit:]

    async def get_ticker(self, symbol: str) -> dict:
        async def _fetch():
            url = f"{self.base_url}/v2/tickers/{symbol}"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, headers={"User-Agent": "forexbot/1.0"})
                resp.raise_for_status()
                return resp.json().get("result") or {}
        return await self._cached(f"ticker:{symbol}", 3.0, _fetch)

    async def get_funding_and_oi(self, symbol: str) -> dict:
        """Perpetual funding rate + open interest, parsed off the SAME cached ticker
        `get_ticker()` already fetches (3s TTL) — no extra network call.

        Field names confirmed against a live testnet /v2/tickers/{symbol} response:
        funding_rate, oi, oi_change_usd_6h, spot_price, mark_price.
        """
        t = await self.get_ticker(symbol)

        def _f(key):
            v = t.get(key)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        return {
            "funding_rate": _f("funding_rate"),
            "oi": _f("oi"),
            "oi_change_6h": _f("oi_change_usd_6h"),
            "spot_price": _f("spot_price"),
            "mark_price": _f("mark_price"),
        }

    async def get_orderbook(self, symbol: str) -> dict:
        """L2 book: {'buy': [bids, price-descending], 'sell': [asks, price-ascending]}.

        Needed because mark price is NOT what you can trade at — on the testnet the
        ask side routinely sits percent(s) away from mark, so a market order to close
        fills nowhere near the mark-based PnL the dashboard shows.
        """
        async def _fetch():
            url = f"{self.base_url}/v2/l2orderbook/{symbol}"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, headers={"User-Agent": "forexbot/1.0"})
                resp.raise_for_status()
                return resp.json().get("result") or {}
        return await self._cached(f"book:{symbol}", 3.0, _fetch)

    async def get_product(self, symbol: str) -> dict:
        """Fetch full product metadata for a symbol."""
        url = f"{self.base_url}/v2/products/{symbol}"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers={"User-Agent": "forexbot/1.0"})
            resp.raise_for_status()
            return resp.json().get("result") or {}

    async def get_product_id(self, symbol: str) -> Optional[int]:
        """Resolve a symbol to its numeric Delta product_id (cached)."""
        if symbol in self._product_cache:
            return self._product_cache[symbol]
        result = await self.get_product(symbol)
        pid = result.get("id")
        if pid is not None:
            self._product_cache[symbol] = pid
        return pid

    async def get_contract_value(self, symbol: str) -> float:
        """Contract value (underlying units per contract) used to scale PnL to USD (cached)."""
        if symbol in self._cv_cache:
            return self._cv_cache[symbol]
        try:
            result = await self.get_product(symbol)
            cv = float(result.get("contract_value") or 0.001)
        except Exception:
            cv = 0.001
        self._cv_cache[symbol] = cv
        return cv

    async def get_fills(self, page_size: int = 200) -> list[dict]:
        """Fetch executed trade fills (real trade history) for the account (cached)."""
        async def _fetch():
            path = "/v2/fills"
            query = f"?page_size={page_size}"
            headers = self._sign("GET", path, query=query)
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"{self.base_url}{path}{query}", headers=headers)
                resp.raise_for_status()
                return resp.json().get("result", []) or []
        return await self._cached(f"fills:{page_size}", 15.0, _fetch)

    async def place_order(
        self,
        symbol: str,
        side: str,          # "buy" or "sell"
        size: int,
        order_type: str = "market_order",
        limit_price: Optional[float] = None,
        reduce_only: bool = False,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> dict:
        """
        Place an order. Optionally attach a 1:2 bracket (stop-loss + take-profit).
        Brackets only attach when opening from a flat position (Delta requirement).
        """
        path = "/v2/orders"
        product_id = await self.get_product_id(symbol)
        payload_dict: dict = {
            "product_id": product_id,
            "product_symbol": symbol,
            "size": int(size),
            "side": side,
            "order_type": order_type,
        }
        if limit_price:
            payload_dict["limit_price"] = str(limit_price)
        if reduce_only:
            payload_dict["reduce_only"] = True
        if stop_loss_price is not None and take_profit_price is not None and not reduce_only:
            sl = f"{stop_loss_price:.1f}"
            tp = f"{take_profit_price:.1f}"
            payload_dict.update({
                "bracket_stop_loss_price": sl,
                "bracket_stop_loss_limit_price": sl,
                "bracket_take_profit_price": tp,
                "bracket_take_profit_limit_price": tp,
            })

        payload_str = json.dumps(payload_dict, separators=(",", ":"))
        headers = self._sign("POST", path, payload=payload_str)
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(f"{self.base_url}{path}", content=payload_str, headers=headers)
            if resp.status_code == 400 and "bracket" in resp.text.lower():
                # bracket rejected (e.g. position exists) -> retry as plain order
                for k in ("bracket_stop_loss_price", "bracket_stop_loss_limit_price",
                          "bracket_take_profit_price", "bracket_take_profit_limit_price"):
                    payload_dict.pop(k, None)
                payload_str = json.dumps(payload_dict, separators=(",", ":"))
                headers = self._sign("POST", path, payload=payload_str)
                resp = await client.post(f"{self.base_url}{path}", content=payload_str, headers=headers)
            resp.raise_for_status()
            self.invalidate_cache()  # position/fills changed
            return resp.json().get("result", {})

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        product_id = await self.get_product_id(symbol)
        path = f"/v2/products/{product_id}/orders/leverage"
        payload_str = json.dumps({"leverage": str(leverage)}, separators=(",", ":"))
        headers = self._sign("POST", path, payload=payload_str)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{self.base_url}{path}", content=payload_str, headers=headers)
            resp.raise_for_status()
            return resp.json().get("result", {})

    async def get_position_size(self, symbol: str) -> float:
        """Signed contracts currently held for the symbol (0 if flat)."""
        try:
            for p in await self.get_positions():
                if p.get("product_symbol") == symbol and p.get("size"):
                    return float(p.get("size") or 0)
        except Exception:
            pass
        return 0.0

    async def get_available_balance(self) -> float:
        try:
            w = await self.get_wallet()
            return float(w.get("available_balance") or w.get("balance") or 0)
        except Exception:
            return 0.0

    async def get_position(self, symbol: str) -> dict:
        """Full position dict for the symbol (or {} if flat)."""
        try:
            for p in await self.get_positions():
                if p.get("product_symbol") == symbol and p.get("size"):
                    return p
        except Exception:
            pass
        return {}

    async def get_live_orders(self, symbol: str) -> list[dict]:
        """Open + pending orders for the symbol (cached 8s, shared across endpoints)."""
        async def _fetch():
            pid = await self.get_product_id(symbol)
            path = "/v2/orders"
            query = f"?product_ids={pid}&states=open,pending"
            headers = self._sign("GET", path, query=query)
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}{path}{query}", headers=headers)
                resp.raise_for_status()
                return resp.json().get("result", []) or []
        return await self._cached(f"orders:{symbol}", 8.0, _fetch)

    async def cancel_order(self, order_id, product_id) -> dict:
        path = "/v2/orders"
        payload = json.dumps({"id": int(order_id), "product_id": int(product_id)}, separators=(",", ":"))
        headers = self._sign("DELETE", path, payload=payload)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.request("DELETE", f"{self.base_url}{path}", content=payload, headers=headers)
            self.invalidate_cache()  # orders changed
            try:
                return resp.json()
            except Exception:
                return {}

    async def place_stop_order(self, symbol: str, side: str, size: int, stop_price: float, kind: str) -> dict:
        """Reduce-only stop order. kind = 'take_profit_order' or 'stop_loss_order'."""
        pid = await self.get_product_id(symbol)
        path = "/v2/orders"
        payload_dict = {
            "product_id": pid, "product_symbol": symbol, "size": int(size), "side": side,
            "order_type": "market_order", "stop_order_type": kind,
            "stop_price": f"{stop_price:.1f}", "reduce_only": True,
            # trigger on MARK price — matches the mark-price chart + the bot's analysis basis
            "stop_trigger_method": "mark_price",
        }
        payload_str = json.dumps(payload_dict, separators=(",", ":"))
        headers = self._sign("POST", path, payload=payload_str)
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(f"{self.base_url}{path}", content=payload_str, headers=headers)
            resp.raise_for_status()
            self.invalidate_cache()  # orders changed
            return resp.json().get("result", {})

    async def get_positions(self) -> list[dict]:
        async def _fetch():
            path = "/v2/positions/margined"
            headers = self._sign("GET", path)
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}{path}", headers=headers)
                resp.raise_for_status()
                return resp.json().get("result", []) or []
        return await self._cached("positions", 12.0, _fetch)

    async def get_wallet(self) -> dict:
        path = "/v2/wallet/balances"
        headers = self._sign("GET", path)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}{path}", headers=headers)
            resp.raise_for_status()
            result = resp.json().get("result", []) or []
            # Prefer the USD/USDT settling balance.
            for item in result:
                if item.get("asset_symbol") in ("USD", "USDT"):
                    return item
            return result[0] if result else {}
