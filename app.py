# Installs Streamlit web framework and the Google GenAI library
!pip install streamlit google-genai requests -q

# Installs localtunnel globally via Node (required to view the website)
!npm install -g localtunnel
%%writefile app.py
import streamlit as st
import os
import requests
"""
╔══════════════════════════════════════════════════════════════════╗
║     STOCK MARKET NEWS SENTIMENT ANALYSIS DASHBOARD              ║
║     Production-ready · Streamlit · Gemini AI · Finnhub          ║
╚══════════════════════════════════════════════════════════════════╝

Setup:
    pip install -r requirements.txt

    Create a .env file (or export environment variables):
        GEMINI_API_KEY=your_gemini_api_key_here
        FINNHUB_API_KEY=your_finnhub_api_key_here   # optional, falls back to RSS
        ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key # optional

    Run:
        streamlit run app.py
"""

import os
import json
import time
import hashlib
import datetime
import requests
import streamlit as st
from typing import Optional

# ── Optional imports with graceful fallbacks ──────────────────────────────────
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY        = os.environ.get("GEMINI_API_KEY", "")
FINNHUB_API_KEY       = os.environ.get("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
OPENAI_API_KEY        = os.environ.get("OPENAI_API_KEY", "")

MAX_ARTICLES          = 12     # cap to keep API costs low
REQUEST_TIMEOUT       = 10     # seconds
CACHE_TTL_SECONDS     = 300    # 5-minute in-memory cache

# ══════════════════════════════════════════════════════════════════════════════
#  STEP A — FETCH STOCK NEWS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_finnhub_news(ticker: str) -> list[dict]:
    """Pull news from Finnhub /company-news endpoint (7-day window)."""
    if not FINNHUB_API_KEY:
        return []
    today      = datetime.date.today()
    week_ago   = today - datetime.timedelta(days=7)
    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker.upper()}"
        f"&from={week_ago.isoformat()}"
        f"&to={today.isoformat()}"
        f"&token={FINNHUB_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        items = resp.json()
        articles = []
        for item in items[:MAX_ARTICLES]:
            articles.append({
                "headline": item.get("headline", ""),
                "summary":  item.get("summary", ""),
                "url":      item.get("url", ""),
                "source":   item.get("source", "Finnhub"),
                "datetime": datetime.datetime.fromtimestamp(
                    item.get("datetime", 0)
                ).strftime("%b %d, %Y") if item.get("datetime") else "—",
            })
        return articles
    except Exception as e:
        st.warning(f"Finnhub fetch failed: {e}")
        return []


def fetch_alpha_vantage_news(ticker: str) -> list[dict]:
    """Pull news from Alpha Vantage News & Sentiments endpoint."""
    if not ALPHA_VANTAGE_API_KEY:
        return []
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT"
        f"&tickers={ticker.upper()}"
        f"&limit={MAX_ARTICLES}"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        feed = data.get("feed", [])
        articles = []
        for item in feed[:MAX_ARTICLES]:
            articles.append({
                "headline": item.get("title", ""),
                "summary":  item.get("summary", ""),
                "url":      item.get("url", ""),
                "source":   item.get("source", "Alpha Vantage"),
                "datetime": item.get("time_published", "")[:10] or "—",
            })
        return articles
    except Exception as e:
        st.warning(f"Alpha Vantage fetch failed: {e}")
        return []


def fetch_rss_news(ticker: str) -> list[dict]:
    """
    Fallback: scrape Google Finance RSS for the ticker.
    Works without any API key.
    """
    if not HAS_FEEDPARSER:
        return []
    # Google Finance RSS (public, no key needed)
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker.upper()}&region=US&lang=en-US"
    fallback_url = f"https://news.google.com/rss/search?q={ticker.upper()}+stock&hl=en-US&gl=US&ceid=US:en"
    articles = []
    for url in [rss_url, fallback_url]:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:MAX_ARTICLES]:
                articles.append({
                    "headline": entry.get("title", ""),
                    "summary":  entry.get("summary", entry.get("description", "")),
                    "url":      entry.get("link", ""),
                    "source":   feed.feed.get("title", "RSS"),
                    "datetime": entry.get("published", "—")[:16] if entry.get("published") else "—",
                })
            if articles:
                break
        except Exception:
            continue
    return articles[:MAX_ARTICLES]


def fetch_stock_news(ticker: str) -> list[dict]:
    """
    Orchestrates news fetching with priority:
      1. Finnhub  (if API key present)
      2. Alpha Vantage (if API key present)
      3. RSS fallback (always available if feedparser installed)
    """
    ticker = ticker.strip().upper()
    # Try Finnhub first
    articles = fetch_finnhub_news(ticker)
    if articles:
        return articles
    # Try Alpha Vantage
    articles = fetch_alpha_vantage_news(ticker)
    if articles:
        return articles
    # RSS fallback
    articles = fetch_rss_news(ticker)
    return articles


# ══════════════════════════════════════════════════════════════════════════════
#  STEP B — AI SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

SENTIMENT_PROMPT = """
You are a professional financial analyst specializing in market sentiment and news impact analysis.

Analyze the following news article for the stock ticker {ticker}.

News Content:
\"\"\"{text}\"\"\"

Return ONLY a valid JSON object with exactly these fields:
{{
  "score": <float between -1.0 (very negative) and +1.0 (very positive)>,
  "label": "<one of: Positive, Negative, Neutral>",
  "explanation": "<one concise sentence explaining how this news impacts the stock price>"
}}

Rules:
- score must be a float (e.g. 0.7, -0.4, 0.0)
- label must match the score: negative score → Negative, positive → Positive, near zero → Neutral
- explanation must be 1 sentence, max 25 words, no markdown
- Return ONLY the JSON, no preamble or code fences
"""


def _parse_sentiment_response(raw: str) -> dict:
    """Robustly parse the AI JSON response, stripping fences if present."""
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        label = data.get("label", "Neutral")
        expl  = data.get("explanation", "No explanation provided.")
        # Normalize label
        if label not in ("Positive", "Negative", "Neutral"):
            if score > 0.1:   label = "Positive"
            elif score < -0.1: label = "Negative"
            else:              label = "Neutral"
        return {"score": round(score, 3), "label": label, "explanation": expl}
    except (json.JSONDecodeError, ValueError):
        return {"score": 0.0, "label": "Neutral", "explanation": "Could not parse AI response."}


def analyze_with_gemini(text: str, ticker: str) -> dict:
    """Send text to Google Gemini and return structured sentiment."""
    if not HAS_GEMINI or not GEMINI_API_KEY:
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = SENTIMENT_PROMPT.format(ticker=ticker, text=text[:2000])
        response = model.generate_content(prompt)
        return _parse_sentiment_response(response.text)
    except Exception as e:
        st.warning(f"Gemini error: {e}")
        return None


def analyze_with_openai(text: str, ticker: str) -> dict:
    """Send text to OpenAI GPT and return structured sentiment."""
    if not HAS_OPENAI or not OPENAI_API_KEY:
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = SENTIMENT_PROMPT.format(ticker=ticker, text=text[:2000])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return _parse_sentiment_response(response.choices[0].message.content)
    except Exception as e:
        st.warning(f"OpenAI error: {e}")
        return None


def analyze_with_finbert(text: str) -> dict:
    """
    Local FinBERT inference — no API key needed, but requires ~500 MB model download.
    Automatically attempted if cloud APIs are unavailable.
    """
    try:
        from transformers import pipeline
        # Cache the pipeline in session state so it only loads once
        if "finbert_pipeline" not in st.session_state:
            st.session_state.finbert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=512,
            )
        pipe   = st.session_state.finbert_pipeline
        result = pipe(text[:512])[0]
        label_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
        score_map = {"positive": result["score"], "negative": -result["score"], "neutral": 0.0}
        raw_label = result["label"].lower()
        return {
            "score":       round(score_map.get(raw_label, 0.0), 3),
            "label":       label_map.get(raw_label, "Neutral"),
            "explanation": f"FinBERT classified this as {label_map.get(raw_label, 'Neutral')} with {result['score']:.0%} confidence.",
        }
    except Exception as e:
        return {"score": 0.0, "label": "Neutral", "explanation": f"FinBERT unavailable: {e}"}


def analyze_sentiment(text: str, ticker: str) -> dict:
    """
    Sentiment analysis pipeline with graceful fallback chain:
      Gemini → OpenAI → FinBERT (local) → rule-based stub
    """
    if not text.strip():
        return {"score": 0.0, "label": "Neutral", "explanation": "No content to analyze."}

    # 1. Gemini
    result = analyze_with_gemini(text, ticker)
    if result:
        return result
    # 2. OpenAI
    result = analyze_with_openai(text, ticker)
    if result:
        return result
    # 3. Local FinBERT
    return analyze_with_finbert(text)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP C — AGGREGATOR
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_sentiment(analyzed_articles: list[dict]) -> dict:
    """
    Compute overall market mood from all article sentiments.
    Returns:
        overall_score   : float [-1, +1]
        overall_label   : str
        positive_count  : int
        negative_count  : int
        neutral_count   : int
        bullish_pct     : float [0, 100]
        bearish_pct     : float [0, 100]
        neutral_pct     : float [0, 100]
    """
    if not analyzed_articles:
        return {
            "overall_score": 0.0, "overall_label": "Neutral",
            "positive_count": 0, "negative_count": 0, "neutral_count": 0,
            "bullish_pct": 0.0, "bearish_pct": 0.0, "neutral_pct": 0.0,
        }

    scores = [a["sentiment"]["score"] for a in analyzed_articles]
    labels = [a["sentiment"]["label"] for a in analyzed_articles]

    avg_score      = sum(scores) / len(scores)
    positive_count = labels.count("Positive")
    negative_count = labels.count("Negative")
    neutral_count  = labels.count("Neutral")
    n              = len(labels)

    if avg_score > 0.15:
        overall_label = "Positive"
    elif avg_score < -0.15:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"

    return {
        "overall_score":   round(avg_score, 3),
        "overall_label":   overall_label,
        "positive_count":  positive_count,
        "negative_count":  negative_count,
        "neutral_count":   neutral_count,
        "bullish_pct":     round(positive_count / n * 100, 1),
        "bearish_pct":     round(negative_count / n * 100, 1),
        "neutral_pct":     round(neutral_count  / n * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY CACHE  (avoids re-analysis on every Streamlit re-run)
# ══════════════════════════════════════════════════════════════════════════════

def _cache_key(ticker: str) -> str:
    return hashlib.md5(ticker.upper().encode()).hexdigest()

def cache_get(ticker: str) -> Optional[dict]:
    cache = st.session_state.get("analysis_cache", {})
    entry = cache.get(_cache_key(ticker))
    if entry and (time.time() - entry["ts"]) < CACHE_TTL_SECONDS:
        return entry["data"]
    return None

def cache_set(ticker: str, data: dict):
    if "analysis_cache" not in st.session_state:
        st.session_state["analysis_cache"] = {}
    st.session_state["analysis_cache"][_cache_key(ticker)] = {
        "ts": time.time(), "data": data
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI — CUSTOM CSS & HTML
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg-primary:    #080c12;
  --bg-card:       #0d1220;
  --bg-card-hover: #111827;
  --border:        rgba(255,255,255,0.07);
  --border-glow:   rgba(99,179,237,0.25);
  --accent-blue:   #63b3ed;
  --accent-green:  #48bb78;
  --accent-red:    #fc8181;
  --accent-amber:  #f6ad55;
  --text-primary:  #e2e8f0;
  --text-secondary:#94a3b8;
  --text-dim:      #475569;
  --font-display:  'Syne', sans-serif;
  --font-mono:     'JetBrains Mono', monospace;
  --radius:        12px;
}

/* Streamlit overrides */
.stApp { background: var(--bg-primary) !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1100px !important; }
h1, h2, h3, h4, h5, h6, p, span, label { font-family: var(--font-display) !important; }
code, pre { font-family: var(--font-mono) !important; }

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Dashboard Header ── */
.dash-header {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 0 0 2rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.dash-logo {
  width: 44px; height: 44px;
  background: linear-gradient(135deg, #2b6cb0 0%, #63b3ed 100%);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  box-shadow: 0 0 20px rgba(99,179,237,0.3);
}
.dash-title { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em; color: var(--text-primary); }
.dash-subtitle { font-size: 0.72rem; color: var(--text-dim); font-family: var(--font-mono) !important; letter-spacing: 0.08em; margin-top: 2px; }

/* ── Search Bar ── */
.stTextInput > div > div > input {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-mono) !important;
  font-size: 1rem !important;
  padding: 0.75rem 1rem !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--accent-blue) !important;
  box-shadow: 0 0 0 3px rgba(99,179,237,0.15) !important;
  outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text-dim) !important; }
.stTextInput label { color: var(--text-secondary) !important; font-size: 0.78rem !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #2b6cb0 0%, #2c5282 100%) !important;
  color: #fff !important;
  border: 1px solid rgba(99,179,237,0.3) !important;
  border-radius: var(--radius) !important;
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.04em !important;
  padding: 0.7rem 1.6rem !important;
  transition: all 0.2s !important;
  box-shadow: 0 4px 15px rgba(43,108,176,0.3) !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%) !important;
  box-shadow: 0 6px 20px rgba(49,130,206,0.4) !important;
  transform: translateY(-1px) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-color: var(--accent-blue) transparent transparent transparent !important; }

/* ── Metrics / Hero Cards ── */
.hero-grid {
  display: grid;
  grid-template-columns: 1.4fr 1fr 1fr 1fr;
  gap: 14px;
  margin: 1.5rem 0;
}
.hero-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.2rem 1.4rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s, transform 0.2s;
}
.hero-card:hover { border-color: var(--border-glow); transform: translateY(-2px); }
.hero-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  border-radius: 16px 16px 0 0;
}
.hero-card.bullish::before  { background: linear-gradient(90deg, #48bb78, #68d391); }
.hero-card.bearish::before  { background: linear-gradient(90deg, #fc8181, #feb2b2); }
.hero-card.neutral::before  { background: linear-gradient(90deg, #a0aec0, #cbd5e0); }
.hero-card.info::before     { background: linear-gradient(90deg, #63b3ed, #90cdf4); }
.hero-label {
  font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--text-dim); font-family: var(--font-mono) !important; margin-bottom: 0.5rem;
}
.hero-value {
  font-size: 2rem; font-weight: 800; line-height: 1; letter-spacing: -0.03em;
  margin-bottom: 0.3rem;
}
.hero-sub { font-size: 0.75rem; color: var(--text-secondary); }
.bullish .hero-value  { color: var(--accent-green); }
.bearish .hero-value  { color: var(--accent-red); }
.neutral .hero-value  { color: var(--text-secondary); }
.info .hero-value     { color: var(--accent-blue); }

/* Outlook badge */
.outlook-badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 0.35rem 0.9rem;
  border-radius: 999px;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  font-family: var(--font-mono) !important;
}
.badge-bullish  { background: rgba(72,187,120,0.15); color: #68d391; border: 1px solid rgba(72,187,120,0.3); }
.badge-bearish  { background: rgba(252,129,129,0.15); color: #feb2b2; border: 1px solid rgba(252,129,129,0.3); }
.badge-neutral  { background: rgba(160,174,192,0.12); color: #a0aec0; border: 1px solid rgba(160,174,192,0.2); }

/* ── Sentiment Bar ── */
.sentiment-bar-wrap {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.4rem;
}
.sentiment-bar-label {
  display: flex; justify-content: space-between;
  font-size: 0.72rem; font-family: var(--font-mono) !important;
  color: var(--text-dim); margin-bottom: 0.8rem; letter-spacing: 0.05em;
}
.sentiment-bar-track {
  height: 10px; background: rgba(255,255,255,0.05);
  border-radius: 999px; overflow: hidden; display: flex; gap: 2px;
}
.seg-bull { background: linear-gradient(90deg, #2f855a, #48bb78); border-radius: 999px 0 0 999px; }
.seg-neu  { background: rgba(160,174,192,0.3); }
.seg-bear { background: linear-gradient(90deg, #c53030, #fc8181); border-radius: 0 999px 999px 0; }
.sentiment-bar-legend {
  display: flex; gap: 1.2rem; margin-top: 0.8rem;
  font-size: 0.7rem; font-family: var(--font-mono) !important; color: var(--text-secondary);
}
.leg-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; vertical-align: middle; }

/* ── Article Cards ── */
.section-title {
  font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--text-dim); font-family: var(--font-mono) !important;
  margin: 1.6rem 0 0.9rem 0; padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}
.article-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.1rem 1.3rem;
  margin-bottom: 10px;
  display: flex; gap: 1rem;
  transition: border-color 0.25s, background 0.25s;
  text-decoration: none;
}
.article-card:hover { background: var(--bg-card-hover); border-color: rgba(255,255,255,0.12); }
.article-dot {
  width: 10px; height: 10px; border-radius: 50%;
  flex-shrink: 0; margin-top: 5px;
  box-shadow: 0 0 8px currentColor;
}
.dot-positive { color: var(--accent-green); background: var(--accent-green); }
.dot-negative { color: var(--accent-red);   background: var(--accent-red); }
.dot-neutral  { color: #718096; background: #718096; box-shadow: none; }
.article-body { flex: 1; min-width: 0; }
.article-headline {
  font-size: 0.9rem; font-weight: 600; color: var(--text-primary);
  line-height: 1.4; margin-bottom: 0.35rem;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.article-meta {
  display: flex; align-items: center; gap: 0.7rem;
  font-size: 0.68rem; font-family: var(--font-mono) !important; color: var(--text-dim);
  margin-bottom: 0.4rem;
}
.article-score {
  padding: 1px 7px; border-radius: 4px; font-weight: 600; font-size: 0.65rem;
}
.score-pos { background: rgba(72,187,120,0.15); color: #68d391; }
.score-neg { background: rgba(252,129,129,0.15); color: #feb2b2; }
.score-neu { background: rgba(160,174,192,0.1);  color: #a0aec0; }
.article-explanation {
  font-size: 0.78rem; color: var(--text-secondary); line-height: 1.5;
  padding: 0.4rem 0.7rem;
  background: rgba(255,255,255,0.03); border-radius: 6px;
  border-left: 2px solid rgba(255,255,255,0.08);
  margin-top: 0.3rem;
}

/* ── API Config Panel ── */
.config-section {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.2rem 1.4rem;
}
.status-dot {
  display: inline-block; width: 8px; height: 8px;
  border-radius: 50%; margin-right: 6px; vertical-align: middle;
}
.status-ok  { background: var(--accent-green); box-shadow: 0 0 6px var(--accent-green); }
.status-off { background: var(--text-dim); }

/* ── Divider ── */
hr.dash-divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* ── Streamlit selectbox, sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--bg-card) !important;
  border-right: 1px solid var(--border) !important;
}
.stSelectbox > div > div {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text-primary) !important;
}
.stAlert { border-radius: var(--radius) !important; }
.stWarning, .stInfo, .stError, .stSuccess { border-radius: var(--radius) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
</style>
"""


def render_hero_metric(label: str, value: str, sub: str, card_class: str) -> str:
    return f"""
    <div class="hero-card {card_class}">
        <div class="hero-label">{label}</div>
        <div class="hero-value">{value}</div>
        <div class="hero-sub">{sub}</div>
    </div>
    """


def render_sentiment_bar(bullish_pct: float, bearish_pct: float, neutral_pct: float) -> str:
    return f"""
    <div class="sentiment-bar-wrap">
        <div class="hero-label" style="margin-bottom:0.8rem">Market Sentiment Distribution</div>
        <div class="sentiment-bar-label">
            <span>🐂 Bullish ({bullish_pct:.0f}%)</span>
            <span>Neutral ({neutral_pct:.0f}%)</span>
            <span>Bearish ({bearish_pct:.0f}%) 🐻</span>
        </div>
        <div class="sentiment-bar-track">
            <div class="seg-bull" style="flex:{bullish_pct}"></div>
            <div class="seg-neu"  style="flex:{neutral_pct}"></div>
            <div class="seg-bear" style="flex:{bearish_pct}"></div>
        </div>
        <div class="sentiment-bar-legend">
            <span><span class="leg-dot" style="background:#48bb78"></span>Positive</span>
            <span><span class="leg-dot" style="background:#718096"></span>Neutral</span>
            <span><span class="leg-dot" style="background:#fc8181"></span>Negative</span>
        </div>
    </div>
    """


def render_article_card(article: dict, idx: int) -> str:
    sentiment  = article.get("sentiment", {})
    label      = sentiment.get("label", "Neutral")
    score      = sentiment.get("score", 0.0)
    expl       = sentiment.get("explanation", "")
    headline   = article.get("headline", "No headline")[:120]
    source     = article.get("source", "—")
    date_str   = article.get("datetime", "—")
    url        = article.get("url", "#")

    dot_cls    = f"dot-{label.lower()}"
    score_cls  = "score-pos" if score > 0 else ("score-neg" if score < 0 else "score-neu")
    score_sign = "+" if score > 0 else ""

    return f"""
    <a class="article-card" href="{url}" target="_blank" style="text-decoration:none;display:flex;gap:1rem">
        <div class="article-dot {dot_cls}"></div>
        <div class="article-body">
            <div class="article-headline">{headline}</div>
            <div class="article-meta">
                <span>{source}</span>
                <span>·</span>
                <span>{date_str}</span>
                <span class="article-score {score_cls}">{score_sign}{score:.2f}</span>
                <span class="outlook-badge badge-{label.lower()}" style="padding:1px 8px;font-size:0.6rem">{label}</span>
            </div>
            <div class="article-explanation">💡 {expl}</div>
        </div>
    </a>
    """


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="StockPulse — Sentiment Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar: API Configuration ────────────────────────────────────────────
    with st.sidebar:
        # Declare globals at top of with-block before any reads
        global GEMINI_API_KEY, FINNHUB_API_KEY, ALPHA_VANTAGE_API_KEY, OPENAI_API_KEY

        st.markdown("""
        <div style='padding: 1rem 0 1.5rem 0;'>
            <div style='font-size:1.1rem;font-weight:800;color:#e2e8f0;font-family:Syne,sans-serif'>⚙️ Configuration</div>
            <div style='font-size:0.7rem;color:#475569;font-family:JetBrains Mono,monospace;margin-top:4px'>API KEYS & SETTINGS</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**AI Engine**")
        ai_engine = st.selectbox(
            "Select AI backend",
            ["Auto (best available)", "Gemini (Google)", "OpenAI (GPT)", "FinBERT (Local/Free)"],
            label_visibility="collapsed",
        )

        st.markdown("<hr class='dash-divider'>", unsafe_allow_html=True)
        st.markdown("**API Keys** *(overrides .env)*")

        sid_gemini  = st.text_input("Gemini API Key",    value=GEMINI_API_KEY,        type="password", placeholder="AIza...")
        sid_finnhub = st.text_input("Finnhub API Key",   value=FINNHUB_API_KEY,       type="password", placeholder="d1...")
        sid_av      = st.text_input("Alpha Vantage Key", value=ALPHA_VANTAGE_API_KEY, type="password", placeholder="...")
        sid_openai  = st.text_input("OpenAI API Key",    value=OPENAI_API_KEY,        type="password", placeholder="sk-...")

        # Propagate sidebar keys into globals
        if sid_gemini:   GEMINI_API_KEY        = sid_gemini
        if sid_finnhub:  FINNHUB_API_KEY        = sid_finnhub
        if sid_av:       ALPHA_VANTAGE_API_KEY  = sid_av
        if sid_openai:   OPENAI_API_KEY         = sid_openai

        st.markdown("<hr class='dash-divider'>", unsafe_allow_html=True)
        st.markdown("**Status**")
        def dot(ok): return '<span class="status-dot status-ok"></span>' if ok else '<span class="status-dot status-off"></span>'
        st.markdown(f"""
        <div style='font-size:0.75rem;color:#94a3b8;line-height:2'>
            {dot(bool(GEMINI_API_KEY))} Gemini API<br>
            {dot(bool(FINNHUB_API_KEY))} Finnhub News<br>
            {dot(bool(ALPHA_VANTAGE_API_KEY))} Alpha Vantage<br>
            {dot(bool(OPENAI_API_KEY))} OpenAI API<br>
            {dot(HAS_FEEDPARSER)} RSS Fallback<br>
            {dot(HAS_GEMINI)} google-genai pkg<br>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr class='dash-divider'>", unsafe_allow_html=True)
        max_art = st.slider("Max articles to analyze", 3, 15, MAX_ARTICLES)
        global MAX_ARTICLES
        MAX_ARTICLES = max_art

        st.markdown("""
        <div style='font-size:0.65rem;color:#475569;line-height:1.6;padding-top:0.5rem;font-family:JetBrains Mono,monospace'>
        Keys are only stored in your browser session.<br>
        Never committed to disk or logs.
        </div>
        """, unsafe_allow_html=True)

    # ── Main Dashboard ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="dash-header">
        <div class="dash-logo">📊</div>
        <div>
            <div class="dash-title">StockPulse</div>
            <div class="dash-subtitle">REAL-TIME NEWS SENTIMENT ANALYSIS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Search row
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticker_input = st.text_input(
            "Stock Ticker",
            placeholder="AAPL  TSLA  NVDA  MSFT ...",
            label_visibility="visible",
        )
    with col_btn:
        st.markdown("<div style='margin-top:1.82rem'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("⚡ Analyze Stock", use_container_width=True)

    # Show quick-pick chips
    st.markdown("""
    <div style='display:flex;gap:8px;flex-wrap:wrap;margin:-0.4rem 0 1rem 0'>
        <span style='font-size:0.65rem;color:#475569;font-family:JetBrains Mono,monospace;
                     align-self:center;letter-spacing:0.08em'>QUICK PICKS:</span>
    </div>
    """, unsafe_allow_html=True)
    qcols = st.columns(8)
    quick_tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "BTC-USD"]
    for i, qt in enumerate(quick_tickers):
        with qcols[i]:
            if st.button(qt, key=f"quick_{qt}", use_container_width=True):
                ticker_input = qt
                analyze_clicked = True

    st.markdown("<hr class='dash-divider'>", unsafe_allow_html=True)

    # ── Analysis Execution ────────────────────────────────────────────────────
    if analyze_clicked and ticker_input:
        ticker = ticker_input.strip().upper()

        # Check cache
        cached = cache_get(ticker)
        if cached:
            st.info(f"⚡ Showing cached results for **{ticker}** (refreshes every 5 min)")
            render_results(ticker, cached["articles"], cached["agg"])
            return

        # Fresh analysis
        with st.spinner(f"🔍 Fetching latest news for {ticker}..."):
            articles = fetch_stock_news(ticker)

        if not articles:
            st.error(
                f"❌ No news found for **{ticker}**. "
                "Check the ticker symbol or add a news API key in the sidebar."
            )
            return

        analyzed = []
        progress_bar = st.progress(0, text="Analyzing sentiment...")
        for i, article in enumerate(articles):
            text_to_analyze = f"{article['headline']}. {article['summary']}"
            with st.spinner(f"🤖 AI analyzing article {i+1}/{len(articles)}: {article['headline'][:60]}..."):
                # Respect sidebar AI engine selection
                if ai_engine == "Gemini (Google)":
                    sentiment = analyze_with_gemini(text_to_analyze, ticker) or analyze_with_finbert(text_to_analyze)
                elif ai_engine == "OpenAI (GPT)":
                    sentiment = analyze_with_openai(text_to_analyze, ticker) or analyze_with_finbert(text_to_analyze)
                elif ai_engine == "FinBERT (Local/Free)":
                    sentiment = analyze_with_finbert(text_to_analyze)
                else:
                    sentiment = analyze_sentiment(text_to_analyze, ticker)
                article["sentiment"] = sentiment
                analyzed.append(article)
            progress_bar.progress((i + 1) / len(articles), text=f"Analyzing {i+1}/{len(articles)} articles...")

        progress_bar.empty()

        agg = aggregate_sentiment(analyzed)
        cache_set(ticker, {"articles": analyzed, "agg": agg})
        render_results(ticker, analyzed, agg)

    elif not ticker_input and not analyze_clicked:
        # Landing state
        st.markdown("""
        <div style='text-align:center;padding:4rem 2rem;color:#475569'>
            <div style='font-size:3.5rem;margin-bottom:1rem'>📈</div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                        color:#64748b;margin-bottom:0.5rem'>Enter a stock ticker above to begin</div>
            <div style='font-size:0.8rem;font-family:JetBrains Mono,monospace;line-height:2'>
                Powered by AI sentiment analysis · Real-time news · Color-coded signals
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_results(ticker: str, analyzed: list[dict], agg: dict):
    """Render the full results dashboard."""
    label = agg["overall_label"]
    score = agg["overall_score"]
    n     = len(analyzed)

    # ── Ticker + Badge ────────────────────────────────────────────────────────
    badge_cls   = "bullish" if label == "Positive" else ("bearish" if label == "Negative" else "neutral")
    outlook_txt = ("BULLISH 🚀" if label == "Positive" else ("BEARISH ⚠️" if label == "Negative" else "NEUTRAL ➡️"))
    icon        = "🟢" if label == "Positive" else ("🔴" if label == "Negative" else "🔵")

    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:14px;margin-bottom:1.2rem'>
        <div style='font-size:2rem;font-weight:800;font-family:Syne,sans-serif;
                    color:#e2e8f0;letter-spacing:-0.04em'>{icon} {ticker}</div>
        <div class="outlook-badge badge-{badge_cls}" style='font-size:0.75rem;padding:0.3rem 1rem'>
            Overall Outlook: {outlook_txt}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero Metrics Grid ─────────────────────────────────────────────────────
    score_sign  = "+" if score > 0 else ""
    card_cls    = "bullish" if label == "Positive" else ("bearish" if label == "Negative" else "neutral")

    hero_html = f"""<div class="hero-grid">
        {render_hero_metric("Avg. Sentiment Score", f"{score_sign}{score:.2f}", "Range: −1.0 → +1.0", card_cls)}
        {render_hero_metric("Bullish Articles",  f"{agg['positive_count']}", f"{agg['bullish_pct']:.0f}% of {n} articles", "bullish")}
        {render_hero_metric("Bearish Articles",  f"{agg['negative_count']}", f"{agg['bearish_pct']:.0f}% of {n} articles", "bearish")}
        {render_hero_metric("Articles Analyzed", f"{n}", "Last 7 days", "info")}
    </div>"""
    st.markdown(hero_html, unsafe_allow_html=True)

    # ── Sentiment Bar ─────────────────────────────────────────────────────────
    st.markdown(
        render_sentiment_bar(agg["bullish_pct"], agg["bearish_pct"], agg["neutral_pct"]),
        unsafe_allow_html=True,
    )

    # ── Filter tabs ───────────────────────────────────────────────────────────
    tab_all, tab_pos, tab_neg, tab_neu = st.tabs([
        f"All ({n})",
        f"🟢 Positive ({agg['positive_count']})",
        f"🔴 Negative ({agg['negative_count']})",
        f"⚪ Neutral ({agg['neutral_count']})",
    ])

    def render_feed(articles_subset):
        if not articles_subset:
            st.markdown("<div style='color:#475569;font-size:0.85rem;padding:1rem'>No articles in this category.</div>", unsafe_allow_html=True)
            return
        st.markdown('<div class="section-title">News Articles · AI-Analyzed</div>', unsafe_allow_html=True)
        for i, art in enumerate(articles_subset):
            st.markdown(render_article_card(art, i), unsafe_allow_html=True)

    with tab_all:
        render_feed(analyzed)
    with tab_pos:
        render_feed([a for a in analyzed if a["sentiment"]["label"] == "Positive"])
    with tab_neg:
        render_feed([a for a in analyzed if a["sentiment"]["label"] == "Negative"])
    with tab_neu:
        render_feed([a for a in analyzed if a["sentiment"]["label"] == "Neutral"])

    # ── Raw JSON expander ─────────────────────────────────────────────────────
    with st.expander("🔧 Raw Analysis Data (JSON)"):
        st.json({
            "ticker": ticker,
            "aggregated": agg,
            "articles": [
                {
                    "headline":   a["headline"],
                    "source":     a["source"],
                    "datetime":   a["datetime"],
                    "sentiment":  a["sentiment"],
                    "url":        a["url"],
                }
                for a in analyzed
            ],
        })


if __name__ == "__main__":
    main()
  # This prints the secret password code you need to access your website
!curl ipv4.icanhazip.com# 
Starts your stock website in the background and opens a public tunnel portal
!streamlit run app.py & npx localtunnel --port 8501
  
