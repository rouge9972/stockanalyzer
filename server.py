import os
import json
import math
import time
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from cachetools import TTLCache

load_dotenv()

app = Flask(__name__)

# ── Groq client (optional) ────────────────────────────────────────────────────
# The market-data endpoints work without a key; only the AI endpoints need one.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Protects your Groq quota (the AI routes are the expensive ones) and softens
# load on Yahoo. In-memory storage is fine for a single-instance free-tier
# deploy; for multi-instance you'd point storage_uri at Redis.
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["300 per hour", "60 per minute"],
    storage_uri="memory://",
)

# ── Caching ───────────────────────────────────────────────────────────────────
# yfinance scrapes Yahoo's unofficial endpoints; from a datacenter IP those get
# rate-limited/blocked quickly. Caching by symbol means repeated views and the
# separate frontend calls (quote/fundamentals/technicals/news for the same
# symbol) hit Yahoo once, not once per request.
_INFO_TTL = int(os.environ.get("CACHE_TTL_INFO", "300"))     # 5 min
_HIST_TTL = int(os.environ.get("CACHE_TTL_HIST", "300"))     # 5 min
_NEWS_TTL = int(os.environ.get("CACHE_TTL_NEWS", "600"))     # 10 min
_SEARCH_TTL = int(os.environ.get("CACHE_TTL_SEARCH", "600")) # 10 min

_info_cache = TTLCache(maxsize=512, ttl=_INFO_TTL)
_fastinfo_cache = TTLCache(maxsize=512, ttl=_INFO_TTL)
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
    """Light retry for transient Yahoo hiccups. Not a fix for a sustained IP
    block — caching and (ultimately) an official API are the real mitigations."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < attempts - 1:
                time.sleep(base_delay * (i + 1))
    raise last


def _fetch_info(symbol: str) -> dict:
    key = symbol.upper()
    cached = _cache_get(_info_cache, key)
    if cached is not None:
        return cached
    info = _with_retry(lambda: yf.Ticker(key).info)
    _cache_put(_info_cache, key, info)
    return info


def _fetch_fast_info(symbol: str) -> dict:
    key = symbol.upper()
    cached = _cache_get(_fastinfo_cache, key)
    if cached is not None:
        return cached

    def pull():
        fi = yf.Ticker(key).fast_info
        return {
            "last_price": fi.last_price,
            "previous_close": fi.previous_close,
            "day_low": fi.day_low,
            "day_high": fi.day_high,
            "year_low": fi.year_low,
            "year_high": fi.year_high,
            "three_month_average_volume": fi.three_month_average_volume,
        }

    data = _with_retry(pull)
    _cache_put(_fastinfo_cache, key, data)
    return data


def _fetch_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    key = f"{symbol.upper()}:{period}"
    cached = _cache_get(_hist_cache, key)
    if cached is not None:
        return cached
    hist = _with_retry(lambda: yf.Ticker(symbol.upper()).history(period=period))
    _cache_put(_hist_cache, key, hist)
    return hist


def _fetch_news(symbol: str) -> list:
    key = symbol.upper()
    cached = _cache_get(_news_cache, key)
    if cached is not None:
        return cached
    raw = _with_retry(lambda: yf.Ticker(key).news) or []
    _cache_put(_news_cache, key, raw)
    return raw


def _fetch_search(q: str) -> list:
    key = q.strip().lower()
    cached = _cache_get(_search_cache, key)
    if cached is not None:
        return cached
    results = _with_retry(lambda: yf.Search(q, max_results=8).quotes) or []
    _cache_put(_search_cache, key, results)
    return results


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


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    """Liveness probe for the host."""
    return jsonify({"status": "ok", "ai_enabled": client is not None})


@app.route("/api/quote/<symbol>")
def quote(symbol):
    try:
        fi = _fetch_fast_info(symbol)
        full_info = _fetch_info(symbol)

        price = _safe(fi.get("last_price"))
        prev_close = _safe(fi.get("previous_close"))
        change = None
        change_pct = None
        if price and prev_close and prev_close != 0:
            change = round(price - prev_close, 4)
            change_pct = round((change / prev_close) * 100, 2)

        return jsonify({
            "symbol": symbol.upper(),
            "name": full_info.get("longName") or full_info.get("shortName") or symbol.upper(),
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "volume": _safe(fi.get("three_month_average_volume")),
            "day_low": _safe(fi.get("day_low")),
            "day_high": _safe(fi.get("day_high")),
            "week52_low": _safe(fi.get("year_low")),
            "week52_high": _safe(fi.get("year_high")),
            "currency": full_info.get("currency", "USD"),
            "exchange": full_info.get("exchange", ""),
            "asset_type": full_info.get("quoteType", ""),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/fundamentals/<symbol>")
def fundamentals(symbol):
    try:
        info = _fetch_info(symbol)
        return jsonify({
            "market_cap": _fmt_large(info.get("marketCap")),
            "pe_ratio": _safe(info.get("trailingPE")),
            "forward_pe": _safe(info.get("forwardPE")),
            "eps": _safe(info.get("trailingEps")),
            "revenue": _fmt_large(info.get("totalRevenue")),
            "profit_margin": _safe(info.get("profitMargins")),
            "beta": _safe(info.get("beta")),
            "dividend_yield": _safe(info.get("dividendYield")),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "employees": info.get("fullTimeEmployees"),
            "description": info.get("longBusinessSummary", ""),
            "week52_change": _safe(info.get("52WeekChange")),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/technicals/<symbol>")
def technicals(symbol):
    try:
        hist = _fetch_history(symbol, period="1y")
        if hist.empty:
            return jsonify({"error": "No historical data"}), 404
        return jsonify(_compute_technicals(hist))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/news/<symbol>")
def news(symbol):
    try:
        raw = _fetch_news(symbol)
        items = []
        for n in raw[:15]:
            c = n.get("content", n)  # new yfinance nests data under "content"
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
        return jsonify(items)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


@app.route("/api/search")
@limiter.limit("120 per minute")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    try:
        results = _fetch_search(q)
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
        return jsonify(items)
    except Exception:
        return jsonify([])


@app.route("/api/portfolio", methods=["POST"])
@limiter.limit("15 per hour; 4 per minute")
def portfolio():
    if client is None:
        return jsonify({"error": "AI portfolio builder unavailable: GROQ_API_KEY not configured."}), 503
    try:
        data = request.get_json()
        capital = data.get("capital", 10000)
        positions = data.get("positions")  # None means AI decides
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

        # Normalise allocations to exactly 100
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
        hist = _fetch_history(s, period="1y")
        info = _fetch_info(s)
        fi = _fetch_fast_info(s)

        price = _safe(fi.get("last_price")) or _safe(info.get("currentPrice")) or _safe(info.get("regularMarketPrice"))
        prev_close = _safe(fi.get("previous_close"))
        change, change_pct = None, None
        if price and prev_close and prev_close != 0:
            change = round(price - prev_close, 4)
            change_pct = round((change / prev_close) * 100, 2)

        quote_data = {
            "symbol": s,
            "name": info.get("longName") or info.get("shortName") or s,
            "price": price,
            "change": change,
            "change_pct": change_pct,
            "currency": info.get("currency", "USD"),
        }

        tech_data = _compute_technicals(hist) if not hist.empty else {}

        fund_data = {
            "market_cap": _fmt_large(info.get("marketCap")),
            "pe_ratio": _safe(info.get("trailingPE")),
            "eps": _safe(info.get("trailingEps")),
            "beta": _safe(info.get("beta")),
            "sector": info.get("sector", "N/A"),
            "dividend_yield": _safe(info.get("dividendYield")),
        }

        raw_news = _fetch_news(s)
        news_data = [
            {"title": n.get("title", ""), "publisher": n.get("publisher", ""), "link": n.get("link", "#")}
            for n in raw_news[:8]
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
    # Local dev only. In production the Procfile runs this under gunicorn,
    # so this block is not used and debug stays off unless you opt in.
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
