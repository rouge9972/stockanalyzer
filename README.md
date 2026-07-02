# StockAnalyzer

A securities-analysis web app (Python / Flask). It pulls live market data,
fundamentals, technical indicators, and news for a ticker, then uses an LLM
(Llama 3.3 70B via Groq) to generate a structured trade thesis, entry/exit
signals, and constraint-based portfolio allocations.

**Not financial advice.** All AI output is illustrative and unvalidated.

## Features
- Real-time quotes, fundamentals, and news (Yahoo Finance via `yfinance`)
- Technical indicators computed locally: RSI-14, MACD (12/26/9), Bollinger
  Bands (20, 2σ), SMA 20/50/200, volume ratio, golden-cross / MACD signals
- LLM analysis endpoint returning a structured directional thesis with
  entry/exit signals (`/api/analyze`)
- LLM portfolio builder that allocates capital across holdings to constraints
  (`/api/portfolio`)
- Symbol search autocomplete

## Honest limitations
- **Market data is single-source.** Everything (prices, fundamentals,
  technicals, news, search) comes from Yahoo via `yfinance`. Groq is the LLM,
  not a market-data source.
- **`yfinance` reliability on cloud hosts.** `yfinance` scrapes Yahoo's
  unofficial endpoints. Yahoo rate-limits/blocks datacenter IPs (the kind cloud
  hosts run on), so the deployed app can intermittently return empty data or
  429s even when it works locally. Mitigations here: in-memory caching by symbol
  and light retry. The durable fix is an official API (Alpha Vantage, Finnhub,
  FMP, Twelve Data) as the price/fundamentals source.
- **AI signals are unvalidated.** No backtest sits behind the BUY/SELL,
  stop-loss, or take-profit output. Treat it as generated commentary.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then paste your Groq key into .env
python server.py                 # http://127.0.0.1:5000
```
The market-data endpoints work without a key; only `/api/analyze` and
`/api/portfolio` need `GROQ_API_KEY`.

## Deploy (public link)

### Render
1. Push this folder to its own GitHub repo (see below).
2. Render → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
   Start command: `gunicorn --workers 1 --threads 4 --timeout 60 server:app`
   (or leave blank — the `Procfile` sets it).
4. Add environment variable `GROQ_API_KEY`.
5. Deploy. Free instances sleep when idle, so the first hit after a pause is a
   slow cold start — fine for a demo link.

### Railway
1. Push to GitHub.
2. Railway → New Project → Deploy from GitHub repo.
3. It auto-detects Python and uses the `Procfile`.
4. Add variable `GROQ_API_KEY`.
5. Generate a public domain under Settings → Networking.

## Turn this folder into its own repo
```bash
cd stockanalyzer
git init
git add .
git commit -m "StockAnalyzer: deploy-ready Flask app"
git branch -M main
git remote add origin https://github.com/<you>/stockanalyzer.git
git push -u origin main
```

## Config
| Variable          | Purpose                          | Default |
|-------------------|----------------------------------|---------|
| `GROQ_API_KEY`    | Enables the AI endpoints         | (unset) |
| `PORT`            | Port to bind (host injects this) | 5000    |
| `FLASK_DEBUG`     | `1` enables debug locally        | 0 (off) |
| `CACHE_TTL_INFO`  | Fundamentals/quote cache seconds | 300     |
| `CACHE_TTL_HIST`  | Price-history cache seconds      | 300     |
| `CACHE_TTL_NEWS`  | News cache seconds               | 600     |
| `CACHE_TTL_SEARCH`| Search cache seconds             | 600     |
