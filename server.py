import os
import json
import math
import time
import logging
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from cachetools import TTLCache

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("stockanalyzer")

app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# DATA_SOURCE decides the PRIMARY market-data backend:
#   "finnhub" -> Finnhub primary, Yahoo (yfinance) as automatic fallback  [default]
#   "yahoo"   -> Yahoo only (personal-use mode); Finnhub code stays dormant
# Flip it by editing one line in .env (local) or one host variable (deployed).
DATA_SOURCE = os.environ.get("DATA_SOURCE", "finnhub").strip().lower()
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "").strip()
FINNHUB_BASE = "https://finnhub.io/api/v1"

# Groq client (optional): the market-data endpoints work without it; only the
# AI endpoints (/api/analyze, /api/portfolio) need it.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def _use_finnhub() -> bool:
    """Finnhub is the primary source only when selected AND a key is present.
    If finnhub is selected but no key is set, we degrade gracefully to Yahoo."""
    return DATA_SOURCE == "finnhub" and bool(FINNHUB_API_KEY)


if DATA_SOURCE == "finnhub" and not FINNHUB_API_KEY:
    log.warning("DATA_SOURCE=finnhub but FINNHUB_API_KEY is empty -> falling back to Yahoo.")

# ── Rate limiting ─────────────────────────────────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["300 per hour", "60 per minute"],
    storage_uri="memory://",
)

# ── Caching ───────────────────────────────────────────────────────────────────
_INFO_TTL = int(os.environ.get("CACHE_TTL_INFO", "300"))
_HIST_TTL = int(os.environ.get("CACHE_TTL_HIST", "300"))
_NEWS_TTL = int(os.environ.get("CACHE_TTL_NEWS", "600"))
_SEARCH_TTL = int(os.environ.get("CACHE_TTL_SEARCH", "600"))

# Provider-level caches hold the NORMALISED dicts (source-independent shape).
_quote_cache = TTLCache(maxsize=512, ttl=_INFO_TTL)
_fund_cache = TTLCache(maxsize=512, ttl=_INFO_TTL)
_hist_cache = TTLCache(maxsize=512, ttl=_HIST_TTL)
_news_cache = TTLCache(maxsize=512, ttl=_NEWS_TTL)
_search_cache = TTLCache(maxsize=256, ttl=_SEARCH_TTL)
_cache_lock = threading.Lock()

DISCLAIMER = (
    "For educational and illustrative purposes only. Not financial advice. "
    "AI-generated signals are unvalidated and may be inaccurate. Do your own research."
)


def _cache_get(cache, key):
    with _cache_lock:
        return cache.get(key)


def _cache_put(cache, key, value):
    with _cache_lock:
        cache[key] = value


def _with_retry(fn, attempts=2, base_delay=0.8):
    """Light retry for transient hiccups. Not a fix for a sustained IP block."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < attempts - 1:
                time.sleep(base_delay * (i + 1))
    raise last


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe(v):
    """Convert numpy / nan / inf scalars to JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _fmt_large(v):
    if v is None:
        return "N/A"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "N/A"
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"


def _compute_technicals(hist: pd.DataFrame) -> dict:
    close = hist["Close"].squeeze()
    volume = hist["Volume"].squeeze()

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi_series = 100 - 100 / (1 + rs)
    rsi = _safe(rsi_series.iloc[-1])

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    macd_val = _safe(macd_line.iloc[-1])
    signal_val = _safe(signal_line.iloc[-1])
    hist_val = _safe(histogram.iloc[-1])

    # Bollinger Bands (20, 2σ)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = _safe((sma20 + 2 * std20).iloc[-1])
    bb_mid = _safe(sma20.iloc[-1])
    bb_lower = _safe((sma20 - 2 * std20).iloc[-1])

    # SMAs
    sma50 = _safe(close.rolling(50).mean().iloc[-1])
    sma200 = _safe(close.rolling(200).mean().iloc[-1])

    current_price = _safe(close.iloc[-1])

    # Volume ratio vs 20-day average
    avg_vol = volume.rolling(20).mean().iloc[-1]
    vol_ratio = _safe(volume.iloc[-1] / avg_vol) if avg_vol and avg_vol > 0 else None

    # Crossover signals
    golden_cross = None
    if sma50 and sma200:
        golden_cross = sma50 > sma200

    macd_bullish = None
    if macd_val is not None and signal_val is not None:
        macd_bullish = macd_val > signal_val

    # Bollinger Band position label
    bb_position = "N/A"
    if current_price and bb_upper and bb_lower and bb_upper != bb_lower:
        pct = (current_price - bb_lower) / (bb_upper - bb_lower)
        if pct > 0.95:
            bb_position = "Near Upper Band"
        elif pct > 0.5:
            bb_position = "Upper Half"
        elif pct > 0.05:
            bb_position = "Lower Half"
        else:
            bb_position = "Near Lower Band"

    return {
        "rsi": rsi,
        "macd": {"macd": macd_val, "signal": signal_val, "histogram": hist_val},
        "bollinger": {"upper": bb_upper, "mid": bb_mid, "lower": bb_lower, "position": bb_position},
        "sma": {"sma20": bb_mid, "sma50": sma50, "sma200": sma200},
        "volume_ratio": vol_ratio,
        "golden_cross": golden_cross,
        "macd_bullish": macd_bullish,
        "current_price": current_price,
    }


# ── Low-level Yahoo (yfinance) fetchers ───────────────────────────────────────

def _yf_info(symbol: str) -> dict:
    return _with_retry(lambda: yf.Ticker(symbol.upper()).info) or {}


def _yf_fast_info(symbol: str) -> dict:
    def pull():
        fi = yf.Ticker(symbol.upper()).fast_info
        return {
            "last_price": fi.last_price,
            "previous_close": fi.previous_close,
            "day_low": fi.day_low,
            "day_high": fi.day_high,
            "year_low": fi.year_low,
            "year_high": fi.year_high,
            "three_month_average_volume": fi.three_month_average_volume,
        }
    return _with_retry(pull)


def _yf_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    return _with_retry(lambda: yf.Ticker(symbol.upper()).history(period=period))


# ── Low-level Finnhub fetchers ────────────────────────────────────────────────

def _fh_get(path: str, params: dict) -> dict:
    p = dict(params)
    p["token"] = FINNHUB_API_KEY
    r = requests.get(f"{FINNHUB_BASE}{path}", params=p, timeout=10)
    r.raise_for_status()
    return r.json()


def _fh_quote(symbol: str) -> dict:
    return _fh_get("/quote", {"symbol": symbol.upper()})


def _fh_profile(symbol: str) -> dict:
    return _fh_get("/stock/profile2", {"symbol": symbol.upper()})


def _fh_metric(symbol: str) -> dict:
    return _fh_get("/stock/metric", {"symbol": symbol.upper(), "metric": "all"})


def _fh_company_news(symbol: str) -> list:
    to = datetime.now(timezone.utc).date()
    frm = to - timedelta(days=14)
    return _fh_get("/company-news", {
        "symbol": symbol.upper(),
        "from": frm.isoformat(),
        "to": to.isoformat(),
    }) or []


def _fh_search(q: str) -> dict:
    return _fh_get("/search", {"q": q})


# ── Normalised providers: Yahoo ───────────────────────────────────────────────
# Each returns the SAME shape regardless of backend, so routes don't care which
# source served the data.

def yahoo_quote(symbol: str) -> dict:
    fi = _yf_fast_info(symbol)
    info = _yf_info(symbol)
    return {
        "price": _safe(fi.get("last_price")),
        "prev_close": _safe(fi.get("previous_close")),
        "day_low": _safe(fi.get("day_low")),
        "day_high": _safe(fi.get("day_high")),
        "week52_low": _safe(fi.get("year_low")),
        "week52_high": _safe(fi.get("year_high")),
        "volume": _safe(fi.get("three_month_average_volume")),
        "name": info.get("longName") or info.get("shortName") or symbol.upper(),
        "currency": info.get("currency", "USD"),
        "exchange": info.get("exchange", ""),
        "asset_type": info.get("quoteType", ""),
        "source": "yahoo",
    }


def yahoo_fundamentals(symbol: str) -> dict:
    info = _yf_info(symbol)
    return {
        "market_cap_raw": info.get("marketCap"),
        "pe": _safe(info.get("trailingPE")),
        "forward_pe": _safe(info.get("forwardPE")),
        "eps": _safe(info.get("trailingEps")),
        "revenue_raw": info.get("totalRevenue"),
        "profit_margin": _safe(info.get("profitMargins")),
        "beta": _safe(info.get("beta")),
        "dividend_yield": _safe(info.get("dividendYield")),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "employees": info.get("fullTimeEmployees"),
        "description": info.get("longBusinessSummary", ""),
        "week52_change": _safe(info.get("52WeekChange")),
        "source": "yahoo",
    }


def yahoo_news(symbol: str) -> list:
    raw = _with_retry(lambda: yf.Ticker(symbol.upper()).news) or []
    items = []
    for n in raw[:15]:
        c = n.get("content", n)
        pub_raw = c.get("pubDate") or c.get("providerPublishTime") or 0
        ts = 0
        if isinstance(pub_raw, str) and pub_raw:
            try:
                ts = int(datetime.fromisoformat(pub_raw.replace("Z", "+00:00")).timestamp())
            except Exception:
                ts = 0
        elif isinstance(pub_raw, (int, float)):
            ts = int(pub_raw)
        provider = c.get("provider") or {}
        canonical = c.get("canonicalUrl") or c.get("clickThroughUrl") or {}
        items.append({
            "title": c.get("title", ""),
            "publisher": provider.get("displayName") if isinstance(provider, dict) else c.get("publisher", ""),
            "link": canonical.get("url") if isinstance(canonical, dict) else c.get("link", "#"),
            "published": ts,
        })
    return items


def yahoo_search(q: str) -> list:
    results = _with_retry(lambda: yf.Search(q, max_results=8).quotes) or []
    items = []
    for r in results:
        sym = r.get("symbol") or r.get("Symbol", "")
        if not sym:
            continue
        items.append({
            "symbol": sym,
            "name": r.get("longname") or r.get("shortname") or r.get("longName") or r.get("shortName") or sym,
            "asset_type": r.get("quoteType") or r.get("typeDisp") or "",
            "exchange": r.get("exchDisp") or r.get("exchange") or "",
        })
    return items


# ── Normalised providers: Finnhub ─────────────────────────────────────────────
# Return None (or empty) to signal "not covered" so the dispatcher falls back.
# NOTE: Finnhub's FREE tier does not serve historical candles, so there is no
# finnhub_history — technicals always use Yahoo. Some fundamentals fields
# (revenue, employees, long description, sector) aren't on the free tier and
# are left blank rather than guessed. Percentages are converted to Yahoo's
# fraction convention so the frontend formats them the same way.

def finnhub_quote(symbol: str):
    q = _fh_quote(symbol) or {}
    if not q or q.get("c") in (None, 0):
        return None  # invalid/uncovered symbol -> fall back to Yahoo
    p = _fh_profile(symbol) or {}
    m = (_fh_metric(symbol) or {}).get("metric", {}) or {}
    return {
        "price": _safe(q.get("c")),
        "prev_close": _safe(q.get("pc")),
        "day_low": _safe(q.get("l")),
        "day_high": _safe(q.get("h")),
        "week52_low": _safe(m.get("52WeekLow")),
        "week52_high": _safe(m.get("52WeekHigh")),
        "volume": None,  # not exposed cleanly on free tier
        "name": p.get("name") or symbol.upper(),
        "currency": p.get("currency") or "USD",
        "exchange": p.get("exchange") or "",
        "asset_type": "",
        "source": "finnhub",
    }


def finnhub_fundamentals(symbol: str):
    p = _fh_profile(symbol) or {}
    m = (_fh_metric(symbol) or {}).get("metric", {}) or {}
    if not p and not m:
        return None
    mcap = p.get("marketCapitalization")  # in millions
    margin = m.get("netProfitMarginTTM")
    dy = m.get("dividendYieldIndicatedAnnual")
    return {
        "market_cap_raw": (mcap * 1e6) if isinstance(mcap, (int, float)) else None,
        "pe": _safe(m.get("peTTM") or m.get("peBasicExclExtraTTM")),
        "forward_pe": None,  # not reliably on free tier
        "eps": _safe(m.get("epsTTM") or m.get("epsBasicExclExtraItemsTTM")),
        "revenue_raw": None,  # not on free tier -> N/A rather than a guess
        "profit_margin": (margin / 100.0) if isinstance(margin, (int, float)) else None,
        "beta": _safe(m.get("beta")),
        "dividend_yield": (dy / 100.0) if isinstance(dy, (int, float)) else None,
        "sector": "N/A",  # Finnhub gives industry, not a sector taxonomy
        "industry": p.get("finnhubIndustry") or "N/A",
        "employees": None,  # not on free profile2
        "description": "",  # long summary not on free tier
        "week52_change": _safe(m.get("52WeekPriceReturnDaily")),
        "source": "finnhub",
    }


def finnhub_news(symbol: str) -> list:
    raw = _fh_company_news(symbol) or []
    items = []
    for n in raw[:15]:
        items.append({
            "title": n.get("headline", ""),
            "publisher": n.get("source", ""),
            "link": n.get("url", "#"),
            "published": int(n.get("datetime", 0) or 0),
        })
    return items


def finnhub_search(q: str) -> list:
    data = _fh_search(q) or {}
    items = []
    for r in (data.get("result") or [])[:8]:
        sym = r.get("symbol") or ""
        if not sym:
            continue
        items.append({
            "symbol": sym,
            "name": r.get("description") or sym,
            "asset_type": r.get("type") or "",
            "exchange": r.get("displaySymbol") or "",
        })
    return items


# ── Dispatchers (primary + fallback + cache) ──────────────────────────────────

def provider_quote(symbol: str) -> dict:
    key = symbol.upper()
    cached = _cache_get(_quote_cache, key)
    if cached is not None:
        return cached
    result = None
    if _use_finnhub():
        try:
            result = finnhub_quote(key)
        except Exception as e:  # noqa: BLE001
            log.warning("Finnhub quote failed for %s (%s); falling back to Yahoo.", key, e)
    if result is None:
        result = yahoo_quote(key)
    _cache_put(_quote_cache, key, result)
    return result


def provider_fundamentals(symbol: str) -> dict:
    key = symbol.upper()
    cached = _cache_get(_fund_cache, key)
    if cached is not None:
        return cached
    result = None
    if _use_finnhub():
        try:
            result = finnhub_fundamentals(key)
        except Exception as e:  # noqa: BLE001
            log.warning("Finnhub fundamentals failed for %s (%s); falling back to Yahoo.", key, e)
    if result is None:
        result = yahoo_fundamentals(key)
    _cache_put(_fund_cache, key, result)
    return result


def provider_news(symbol: str) -> list:
    key = symbol.upper()
    cached = _cache_get(_news_cache, key)
    if cached is not None:
        return cached
    result = None
    if _use_finnhub():
        try:
            fetched = finnhub_news(key)
            result = fetched if fetched else None
        except Exception as e:  # noqa: BLE001
            log.warning("Finnhub news failed for %s (%s); falling back to Yahoo.", key, e)
    if result is None:
        try:
            result = yahoo_news(key)
        except Exception:
            result = []
    _cache_put(_news_cache, key, result)
    return result


def provider_search(q: str) -> list:
    key = q.strip().lower()
    cached = _cache_get(_search_cache, key)
    if cached is not None:
        return cached
    result = None
    if _use_finnhub():
        try:
            fetched = finnhub_search(q)
            result = fetched if fetched else None
        except Exception as e:  # noqa: BLE001
            log.warning("Finnhub search failed for '%s' (%s); falling back to Yahoo.", q, e)
    if result is None:
        try:
            result = yahoo_search(q)
        except Exception:
            result = []
    _cache_put(_search_cache, key, result)
    return result


def provider_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Always Yahoo: Finnhub's free tier no longer serves historical candles."""
    key = f"{symbol.upper()}:{period}"
    cached = _cache_get(_hist_cache, key)
    if cached is not None:
        return cached
    hist = _yf_history(symbol, period=period)
    _cache_put(_hist_cache, key, hist)
    return hist


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "ok",
        "data_source": "finnhub" if _use_finnhub() else "yahoo",
        "ai_enabled": client is not None,
    })


@app.route("/api/quote/<symbol>")
def quote(symbol):
    try:
        q = provider_quote(symbol)
        price = _safe(q.get("price"))
        prev_close = _safe(q.get("prev_close"))
        change = change_pct = None
        if price and prev_close and prev_close != 0:
            change = round(price - prev_close, 4)
            change_pct = round((change / prev_close) * 100, 2)
        return jsonify({
            "symbol": symbol.upper(),
            "name": q.get("name") or symbol.upper(),
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "volume": _safe(q.get("volume")),
            "day_low": _safe(q.get("day_low")),
            "day_high": _safe(q.get("day_high")),
            "week52_low": _safe(q.get("week52_low")),
            "week52_high": _safe(q.get("week52_high")),
            "currency": q.get("currency", "USD"),
            "exchange": q.get("exchange", ""),
            "asset_type": q.get("asset_type", ""),
            "source": q.get("source"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/fundamentals/<symbol>")
def fundamentals(symbol):
    try:
        f = provider_fundamentals(symbol)
        return jsonify({
            "market_cap": _fmt_large(f.get("market_cap_raw")),
            "pe_ratio": _safe(f.get("pe")),
            "forward_pe": _safe(f.get("forward_pe")),
            "eps": _safe(f.get("eps")),
            "revenue": _fmt_large(f.get("revenue_raw")),
            "profit_margin": _safe(f.get("profit_margin")),
            "beta": _safe(f.get("beta")),
            "dividend_yield": _safe(f.get("dividend_yield")),
            "sector": f.get("sector", "N/A"),
            "industry": f.get("industry", "N/A"),
            "employees": f.get("employees"),
            "description": f.get("description", ""),
            "week52_change": _safe(f.get("week52_change")),
            "source": f.get("source"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/technicals/<symbol>")
def technicals(symbol):
    try:
        hist = provider_history(symbol, period="1y")
        if hist.empty:
            return jsonify({"error": "No historical data"}), 404
        return jsonify(_compute_technicals(hist))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/news/<symbol>")
def news(symbol):
    try:
        return jsonify(provider_news(symbol))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search")
@limiter.limit("120 per minute")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    try:
        return jsonify(provider_search(q))
    except Exception:
        return jsonify([])


@app.route("/api/analyze", methods=["POST"])
@limiter.limit("15 per hour; 4 per minute")
def analyze():
    if client is None:
        return jsonify({"error": "AI analysis unavailable: GROQ_API_KEY not configured."}), 503
    try:
        data = request.get_json()
        symbol = data.get("symbol", "").upper()
        quote_data = data.get("quote", {})
        tech_data = data.get("technicals", {})
        fund_data = data.get("fundamentals", {})
        news_data = data.get("news", [])

        news_headlines = "\n".join(
            f"- [{n.get('publisher','')}] {n.get('title','')}"
            for n in news_data[:8]
        )

        prompt = f"""You are a senior quantitative analyst. Analyze the following data for {symbol} and return a JSON object only — no markdown, no extra text.

QUOTE:
{json.dumps(quote_data, indent=2)}

TECHNICAL INDICATORS:
{json.dumps(tech_data, indent=2)}

FUNDAMENTALS:
{json.dumps(fund_data, indent=2)}

RECENT NEWS:
{news_headlines}

Return this exact JSON structure (all fields required):
{{
  "direction": "bullish" | "bearish" | "neutral",
  "confidence": <integer 0-100>,
  "summary": "<1-2 sentence thesis>",
  "bull_case": ["<reason>", ...],
  "bear_case": ["<reason>", ...],
  "key_catalysts": ["<event to watch>", ...],
  "risks": ["<risk factor>", ...],
  "time_horizon": "<e.g. short-term (1-4 weeks)>",
  "entry_signal": {{
    "action": "BUY" | "SELL" | "WAIT",
    "trigger": "<what technical/fundamental condition justifies this>",
    "suggested_entry": "<price or range>",
    "stop_loss": "<price>",
    "take_profit": "<price>",
    "risk_reward": "<e.g. 2.5:1>"
  }},
  "exit_signal": {{
    "action": "CLOSE" | "HOLD" | "SCALE OUT",
    "trigger": "<condition that would prompt exit>",
    "note": "<any nuance>"
  }}
}}"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1500,
            messages=[
                {"role": "system", "content": "You are a senior quantitative analyst. Always respond with valid JSON only — no markdown fences, no explanations outside the JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        raw = completion.choices[0].message.content.strip()
        result = json.loads(raw)
        result["disclaimer"] = DISCLAIMER
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/portfolio", methods=["POST"])
@limiter.limit("15 per hour; 4 per minute")
def portfolio():
    if client is None:
        return jsonify({"error": "AI portfolio builder unavailable: GROQ_API_KEY not configured."}), 503
    try:
        data = request.get_json()
        capital = data.get("capital", 10000)
        positions = data.get("positions")
        risk = (data.get("risk") or "").strip()
        focus = data.get("focus", [])
        sectors = data.get("sectors", [])
        geography = (data.get("geography") or "").strip()
        currency = data.get("currency", "USD")

        positions_str = str(int(positions)) if positions is not None else "AI's discretion"
        risk_str = risk if risk else "AI's discretion"
        geo_str = geography if geography else "AI's discretion"
        focus_str = ", ".join(focus) if focus else "AI's discretion"
        sectors_str = ", ".join(sectors) if sectors else "AI's discretion"

        prompt = f"""You are a senior portfolio manager. Build an optimal investment portfolio based on these client preferences and return a JSON object only — no markdown, no extra text.
Any preference marked "AI's discretion" means the client has no constraint — use your professional judgement for that dimension.

CLIENT PREFERENCES:
- Capital to invest: {capital} {currency}
- Number of positions: {positions_str}
- Risk tolerance: {risk_str}
- Investment focus: {focus_str}
- Preferred sectors: {sectors_str}
- Geographic focus: {geo_str}

Return this exact JSON structure. Allocations must sum to exactly 100.0:
{{
  "holdings": [
    {{
      "symbol": "<ticker symbol, e.g. AAPL>",
      "name": "<full company/fund name>",
      "sector": "<sector>",
      "asset_type": "<Stock | ETF | Bond | Crypto>",
      "allocation_pct": <number, e.g. 15.0>,
      "allocation_amount": <number, capital * allocation_pct / 100>,
      "reason": "<1 sentence: why this holding fits the portfolio>"
    }}
  ],
  "thesis": "<2-3 sentence overall portfolio rationale>",
  "risk_assessment": "<1-2 sentence risk profile description>",
  "expected_dividend_yield": "<e.g. 2.1%>",
  "sector_breakdown": {{"Technology": 35, "Healthcare": 20}},
  "diversification_note": "<1 sentence on diversification>"
}}"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=2000,
            messages=[
                {"role": "system", "content": "You are a senior portfolio manager. Always respond with valid JSON only — no markdown fences, no explanations outside the JSON. Allocation percentages must sum to exactly 100."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        raw = completion.choices[0].message.content.strip()
        result = json.loads(raw)

        holdings = result.get("holdings", [])
        total = sum(h.get("allocation_pct", 0) for h in holdings)
        if total > 0:
            for h in holdings:
                h["allocation_pct"] = round(h["allocation_pct"] / total * 100, 2)
                h["allocation_amount"] = round(float(capital) * h["allocation_pct"] / 100, 2)

        result["disclaimer"] = DISCLAIMER
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/report/<symbol>")
def report(symbol):
    """Collect all data for a symbol and return a combined report object."""
    try:
        s = symbol.upper()
        q = provider_quote(s)
        f = provider_fundamentals(s)
        hist = provider_history(s, period="1y")

        price = _safe(q.get("price"))
        prev_close = _safe(q.get("prev_close"))
        change = change_pct = None
        if price and prev_close and prev_close != 0:
            change = round(price - prev_close, 4)
            change_pct = round((change / prev_close) * 100, 2)

        quote_data = {
            "symbol": s,
            "name": q.get("name") or s,
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "currency": q.get("currency", "USD"),
        }

        tech_data = _compute_technicals(hist) if not hist.empty else {}

        fund_data = {
            "market_cap": _fmt_large(f.get("market_cap_raw")),
            "pe_ratio": _safe(f.get("pe")),
            "eps": _safe(f.get("eps")),
            "beta": _safe(f.get("beta")),
            "sector": f.get("sector", "N/A"),
            "dividend_yield": _safe(f.get("dividend_yield")),
        }

        news_data = [
            {"title": n.get("title", ""), "publisher": n.get("publisher", ""), "link": n.get("link", "#")}
            for n in provider_news(s)[:8]
        ]

        return jsonify({
            "quote": quote_data,
            "technicals": tech_data,
            "fundamentals": fund_data,
            "news": news_data,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Local dev only. In production the Procfile runs this under gunicorn.
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
