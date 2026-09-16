import axios from "axios";

const CONFIGURED = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Keep the configured PORT, but inherit the hostname the page was actually opened on.
// Windows resolves "localhost" to ::1 before 127.0.0.1, and on this machine
// wslrelay.exe holds [::1]:8000 and resets every connection — so a hard-coded
// localhost here leaves the dashboard rendering with no data whenever the page is
// opened on 127.0.0.1. Inheriting the hostname means the API is reachable however you
// got here: localhost, 127.0.0.1, or the machine's LAN IP.
const BASE = (() => {
  if (typeof window === "undefined") return CONFIGURED;   // SSR: nothing to inherit
  try {
    const port = new URL(CONFIGURED).port || "8000";
    return `${window.location.protocol}//${window.location.hostname}:${port}`;
  } catch {
    return CONFIGURED;
  }
})();

export const api = axios.create({ baseURL: BASE, timeout: 12000 });

// Retry ONLY true network errors (backend unreachable / timeout) once, briefly.
// Do NOT retry 5xx — that would amplify load when the server is busy.
api.interceptors.response.use(undefined, async (error) => {
  const cfg: any = error?.config;
  const networkError = !error?.response;  // no HTTP response at all
  if (cfg && networkError && !cfg.__retried) {
    cfg.__retried = true;
    await new Promise((r) => setTimeout(r, 800));
    return api(cfg);
  }
  return Promise.reject(error);
});

// Bot controls
export const startBot  = () => api.post("/bot/start");
export const stopBot   = () => api.post("/bot/stop");
export const getBotStatus = () => api.get("/bot/status");
export const setBotSymbol = (symbol: string) => api.post(`/bot/symbol?symbol=${symbol}`);

// Trade data
export const getTradeLogs = (limit = 50) => api.get(`/trades/logs?limit=${limit}`);
export const getTradeStats = () => api.get("/trades/stats");
export const getPnl = (symbol?: string) => api.get(`/trades/pnl${symbol ? `?symbol=${symbol}` : ""}`);
export const getOrders = (limit = 200, symbol?: string, offset = 0) =>
  api.get(`/trades/orders?limit=${limit}&offset=${offset}${symbol ? `&symbol=${symbol}` : ""}`);
export const getAllPositions = () => api.get("/trades/positions");
export const closePosition = (symbol: string, force = false) =>
  api.post(`/trades/close?symbol=${symbol}${force ? "&force=true" : ""}`);
export const updateStopLoss = (symbol: string, price: number) =>
  api.post(`/trades/sl?symbol=${symbol}&price=${price}`);
export const runBacktest = (symbol?: string, bars = 1500) =>
  api.get(`/bot/backtest?bars=${bars}${symbol ? `&symbol=${symbol}` : ""}`, { timeout: 90000 });
export const getPerformance = () => api.get("/bot/performance");
// limit is the history depth the Training Monitor pages through as you scroll.
export const getTraining = (days = 30, limit = 60) =>
  api.get(`/bot/training?days=${days}&limit=${limit}`);

// News + economic calendar
export const getNewsAll = () => api.get("/news/all");
// refresh=true is the call that spends AI usage; the plain GET reads the cache.
// Generation runs through the Claude CLI and takes ~40s, so allow well beyond that —
// the backend's own cap is AI_TIMEOUT_SEC (150s).
export const getBrief = (refresh = false) =>
  api.get(`/news/brief${refresh ? "?refresh=true" : ""}`, { timeout: refresh ? 180000 : 12000 });

// Market
export const getTicker  = (symbol?: string) => api.get(`/market/ticker${symbol ? `?symbol=${symbol}` : ""}`);
export const getCandles = (symbol?: string, resolution = 5, limit = 250) =>
  api.get(`/market/candles?resolution=${resolution}&limit=${limit}${symbol ? `&symbol=${symbol}` : ""}`);
export const getIndicators = (symbol?: string, resolution = 5, limit = 250) =>
  api.get(`/market/indicators?resolution=${resolution}&limit=${limit}${symbol ? `&symbol=${symbol}` : ""}`);
export const getWallet    = () => api.get("/market/wallet");
export const getPositions = () => api.get("/market/positions");
