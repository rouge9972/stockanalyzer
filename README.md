# StockAnalyzer

A securities-analysis web app (Python / Flask). It pulls live market data,
fundamentals, technical indicators, and news for a ticker, then uses an LLM
(Llama 3.3 70B via Groq) to generate a structured trade thesis, entry/exit
signals, and constraint-based portfolio allocations.

**Not financial advice.** All AI output is illustrative and unvalidated.

## Data sources & the DATA_SOURCE switch
Market data has two backends, chosen by the `DATA_SOURCE` environment variable:

- `DATA_SOURCE=finnhub` (default) — **Finnhub is primary, Yahoo is automatic
  fallback.** Recommended for a public/deployed link. Finnhub is an official
  keyed API, so it isn't IP-blocked the way scraping is.
- `DATA_SOURCE=yahoo` — **Yahoo only** (personal-use mode). Finnhub code stays
  in place but dormant.

Flip modes by editing one line in `.env` (local) or one variable in the host
dashboard (deployed). Both code paths always exist; nothing is deleted.

### Who serves what in `finnhub` mode
| Data                              | Source            |
|-----------------------------------|-------------------|
| Live quote (price, day range)     | Finnhub           |
| Core fundamentals (P/E, EPS, beta, margin, div yield, market cap, 52-wk) | Finnhub |
| Company news                      | Finnhub           |
| Symbol search                     | Finnhub           |
| **Technical-indicator history**   | **Yahoo** (see below) |
| Any symbol Finnhub doesn't cover  | Yahoo (fallback)  |

## Honest limitations
- **Technicals still depend on Yahoo.** Finnhub's *free* tier no longer serves
  historical price candles, and RSI/MACD/Bollinger/SMA need a year of daily
  closes. So even in `finnhub` mode the technicals chart is computed from
  yfinance data and carries yfinance's cloud-IP reliability risk. Quotes,
  fundamentals, news, and search do not — those are Finnhub.
- **Some fundamentals fields are blank in `finnhub` mode.** Revenue, employee
  count, sector taxonomy, and the long company description aren't on Finnhub's
  free tier. They're left as N/A rather than guessed. Switch to `yahoo` mode
  locally if you want the richer descriptive fields.
- **AI signals are unvalidated.** No backtest sits behind the BUY/SELL,
  stop-loss, or take-profit output. Treat it as generated commentary.
- **First deploy sanity check:** open a well-known ticker (e.g. AAPL) and
  eyeball P/E, margin, and dividend yield against any public source. Finnhub
  percentages are converted to Yahoo's fraction convention; if any ratio looks
  off by ~100x, that field's conversion needs a tweak.

## Environment variables
| Variable          | Purpose                                    | Default   |
|-------------------|--------------------------------------------|-----------|
| `DATA_SOURCE`     | `finnhub` (primary+fallback) or `yahoo`    | finnhub   |
| `FINNHUB_API_KEY` | Finnhub key; required when `finnhub` mode  | (unset)   |
| `GROQ_API_KEY`    | Enables the AI endpoints                    | (unset)   |
| `PORT`            | Port to bind (host injects this)            | 5000      |
| `FLASK_DEBUG`     | `1` enables debug locally                   | 0 (off)   |
| `CACHE_TTL_INFO`  | Quote/fundamentals cache seconds            | 300       |
| `CACHE_TTL_HIST`  | Price-history cache seconds                 | 300       |
| `CACHE_TTL_NEWS`  | News cache seconds                          | 600       |
| `CACHE_TTL_SEARCH`| Search cache seconds                        | 600       |

If `DATA_SOURCE=finnhub` but no key is set, the app logs a warning and runs in
Yahoo mode — it won't crash.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # paste your Finnhub + Groq keys into .env
python server.py                 # http://127.0.0.1:5000
```

## Deploy (public link)

### Render
1. Push this folder to its own GitHub repo.
2. Render -> New -> Web Service -> connect the repo.
3. Build command: `pip install -r requirements.txt`
   Start command: leave blank (the `Procfile` sets gunicorn).
4. Add environment variables: `DATA_SOURCE=finnhub`, `FINNHUB_API_KEY`, `GROQ_API_KEY`.
5. Deploy. Free instances sleep when idle; the first hit after a pause is a slow
   cold start — fine for a demo link.

### Railway
1. Push to GitHub.
2. Railway -> New Project -> Deploy from GitHub repo (auto-detects the Procfile).
3. Add variables: `DATA_SOURCE`, `FINNHUB_API_KEY`, `GROQ_API_KEY`.
4. Settings -> Networking -> generate a public domain.

## Health check
`GET /healthz` returns the active data source and whether AI is enabled — handy
for confirming your env vars took effect after deploy.
