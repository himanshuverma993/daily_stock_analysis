# -*- coding: utf-8 -*-
"""Macro-Sentiment & Risk Engine — hands-off JSON producer for Genie Trader Pro.

Pipeline (every run, zero human intervention):

1. ``collect_market_snapshot`` — global risk barometers via yfinance
   (indices, VIX, DXY, US 10Y yield, WTI, gold, BTC). Free, no API key.
2. ``collect_headlines`` — recent macro headlines. Tavily when
   ``TAVILY_API_KEYS`` is configured, otherwise Google News RSS (no key).
3. ``generate_with_llm`` — an LLM (first available of
   Gemini -> OpenAI-compatible -> Anthropic -> DeepSeek -> LiteLLM env) is
   forced by the system prompt to answer with ONE strict-English JSON object.
4. ``_rule_based_payload`` — deterministic fallback that always produces a
   valid payload from market data alone, so the output file is *never*
   missing, even with no LLM key, no network, or a malformed LLM response.
5. ``write_payload`` — atomic write of ``sentiment_latest.json``.

Output contract (exactly these keys, exactly this order):

.. code-block:: json

    {
      "market_sentiment": "bullish" | "bearish" | "neutral",
      "macro_score": 0.85,
      "timestamp_utc": "ISO-8601",
      "hermes_risk_feed": {
        "summary": "2-sentence English macro summary.",
        "black_swan_warnings": ["high-impact risks for Hermes risk overrides"]
      },
      "teacher_brain_catalysts": {
        "key_drivers": ["stable catalyst tokens for SQLite pattern matching"]
      }
    }

Usage::

    python main.py --macro-sentiment
    python -m src.macro_sentiment.engine            # standalone
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENGINE_VERSION = "1.0.0"
OUTPUT_FILENAME = "sentiment_latest.json"

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------
# Strict-English system prompt (mandated for every LLM face in this repo).
# ----------------------------------------------------------------------------

STRICT_ENGLISH_DIRECTIVE = (
    "Analyze the data, translate all context, and generate the final output "
    "STRICTLY in English."
)

SYSTEM_PROMPT = f"""You are the Macro-Sentiment and Risk Engine of an autonomous algorithmic \
trading system (Genie Trader Pro).

GLOBAL DIRECTIVE (highest priority): {STRICT_ENGLISH_DIRECTIVE} Translate any non-English \
source material into English; never emit Chinese or any other non-English text in any value.

Your task: given a snapshot of global market data and recent macro headlines, output ONE \
valid JSON object — and nothing else (no markdown fences, no commentary, no extra keys).

Output contract (exact keys, exactly these, valid JSON):

{{
  "market_sentiment": "bullish" | "bearish" | "neutral",
  "macro_score": <float from -1.0 (extreme bear) to 1.0 (extreme bull)>,
  "hermes_risk_feed": {{
    "summary": "<exactly two sentences of English macro summary>",
    "black_swan_warnings": ["<specific high-impact risks that should trigger risk overrides \
(widen ATR stops or halt trading)>"]
  }},
  "teacher_brain_catalysts": {{
    "key_drivers": ["<3-8 short stable catalyst tokens>"]
  }}
}}

Calibration rules you must respect:
- macro_score 1.0 = extreme bull (euphoric risk-on), 0.0 = macro neutral, -1.0 = extreme bear \
(panic / systemic risk-off). Use two decimal places.
- VIX >= 32 or an equity index daily move <= -3% must be reflected as clearly bearish and \
must appear in black_swan_warnings.
- 10Y UST yield >= 4.5% is a tightening headwind; >= 4.8% is black-swan grade.
- black_swan_warnings must be specific, dated events or conditions (e.g. "FOMC decision \
within 24h", "10Y UST yield 4.85% — tightening shock", "US-China tariff escalation"), never \
generic filler. Empty list [] is allowed when genuinely nothing qualifies.
- key_drivers must be short stable noun-tokens optimised for SQLite pattern matching by the \
trading system's TeacherBrain (good: "FOMC rate decision", "US CPI surprise", "OPEC supply \
cut", "NVIDIA earnings", "US-China tariff escalation"). Plain English, no sentences.
- hermes_risk_feed.summary is exactly two sentences, at most ~60 words total.
"""

# ----------------------------------------------------------------------------
# Universe of risk barometers (Yahoo Finance symbols — free, no key).
# ----------------------------------------------------------------------------

MARKET_INSTRUMENTS: List[Dict[str, str]] = [
    {"symbol": "^GSPC", "name": "S&P 500", "role": "equity"},
    {"symbol": "^IXIC", "name": "Nasdaq Composite", "role": "equity"},
    {"symbol": "^DJI", "name": "Dow Jones Industrials", "role": "equity"},
    {"symbol": "^FTSE", "name": "FTSE 100", "role": "equity"},
    {"symbol": "^N225", "name": "Nikkei 225", "role": "equity"},
    {"symbol": "^HSI", "name": "Hang Seng", "role": "equity"},
    {"symbol": "000001.SS", "name": "SSE Composite", "role": "equity"},
    {"symbol": "^VIX", "name": "CBOE VIX", "role": "volatility"},
    {"symbol": "DX-Y.NYB", "name": "US Dollar Index (DXY)", "role": "fx"},
    {"symbol": "^TNX", "name": "US 10Y Treasury Yield", "role": "rates"},
    {"symbol": "CL=F", "name": "WTI Crude Oil", "role": "commodity"},
    {"symbol": "GC=F", "name": "Gold Futures", "role": "commodity"},
    {"symbol": "BTC-USD", "name": "Bitcoin", "role": "crypto"},
]

# Google News RSS queries (free, key-less). English locale is forced so
# TeacherBrain/Hermes only ever receive English tokens.
NEWS_RSS_QUERIES: List[str] = [
    "federal reserve interest rate FOMC",
    "US inflation CPI jobs report economy",
    "tariffs trade war sanctions economy",
    "oil OPEC energy supply shock",
    "stock market selloff recession credit risk",
    "China economy stimulus property market",
]

_NEWS_RSS_URL = (
    "https://news.google.com/rss/search?q={query}+when:2d&hl=en-US&gl=US&ceid=US:en"
)

_HEADLINE_RISK_KEYWORDS = (
    "default", "collapse", "crash", "crisis", "war", "invasion", "nuclear",
    "sanction", "embargo", "circuit breaker", "trading halt", "emergency",
    "bailout", "contagion", "bank run", "insolvency", "bankruptcy",
    "credit event", "black swan", "plunge", "panic",
)

_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")


@dataclass
class EngineContext:
    """Raw material for one engine run."""

    snapshot: List[Dict[str, Any]] = field(default_factory=list)
    snapshot_errors: List[str] = field(default_factory=list)
    headlines: List[Dict[str, str]] = field(default_factory=list)
    news_source: str = "google-news-rss"
    previous_score: Optional[float] = None
    llm_used: bool = False
    llm_model: Optional[str] = None
    fallback_reason: Optional[str] = None


# ----------------------------------------------------------------------------
# Time / IO helpers
# ----------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _to_float(value: Any) -> Optional[float]:
    """Best-effort numeric coercion; never raises (fallback safety net)."""
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_output_path(output_path: Optional[str] = None) -> Path:
    """Resolve the JSON output path (CLI arg > env > repo-root default)."""
    raw = (output_path or os.getenv("MACRO_SENTIMENT_OUTPUT") or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (_REPO_ROOT / path)
    return _REPO_ROOT / OUTPUT_FILENAME


def load_previous_score(output_path: Path) -> Optional[float]:
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        score = data.get("macro_score")
        if isinstance(score, (int, float)):
            return _clamp(round(float(score), 2))
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# Step 1: market snapshot (yfinance — free, no key; fully optional)
# ----------------------------------------------------------------------------

def collect_market_snapshot(ctx: EngineContext) -> None:
    try:
        import yfinance as yf  # lazy: heavy import, optional for unit tests
    except Exception as exc:  # pragma: no cover - environment dependent
        ctx.snapshot_errors.append(f"yfinance unavailable: {exc}")
        logger.warning("yfinance import failed: %s", exc)
        return

    for spec in MARKET_INSTRUMENTS:
        symbol = spec["symbol"]
        try:
            hist = yf.Ticker(symbol).history(period="5d", interval="1d")
            if hist is None or hist.empty or "Close" not in hist:
                raise ValueError("empty history")
            closes = [float(v) for v in hist["Close"].dropna().tolist()]
            if not closes:
                raise ValueError("no closes")
            last = closes[-1]
            prev = closes[-2] if len(closes) > 1 else None
            pct = ((last - prev) / prev * 100.0) if prev else None
            ctx.snapshot.append(
                {
                    "symbol": symbol,
                    "name": spec["name"],
                    "role": spec["role"],
                    "last": round(last, 4),
                    "change_pct_1d": round(pct, 2) if pct is not None else None,
                }
            )
        except Exception as exc:
            ctx.snapshot_errors.append(f"{symbol}: {exc}")
            logger.info("Market snapshot fetch failed for %s: %s", symbol, exc)
    logger.info(
        "Market snapshot: %d/%d instruments OK",
        len(ctx.snapshot),
        len(MARKET_INSTRUMENTS),
    )


# ----------------------------------------------------------------------------
# Step 2: macro headlines (Tavily if keyed, else Google News RSS)
# ----------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 15) -> str:
    import requests

    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "MacroSentimentEngine/1.0 (+github-actions)"},
    )
    resp.raise_for_status()
    return resp.text


def _collect_rss_headlines(ctx: EngineContext) -> None:
    from urllib.parse import quote_plus

    seen = set()
    for query in NEWS_RSS_QUERIES:
        url = _NEWS_RSS_URL.format(query=quote_plus(query))
        try:
            payload = _http_get(url)
            root = ET.fromstring(payload)
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                source = (getattr(item.find("source"), "text", "") or "").strip()
                published = (item.findtext("pubDate") or "").strip()
                key = title.lower()
                if not title or key in seen:
                    continue
                seen.add(key)
                ctx.headlines.append(
                    {"title": title, "source": source, "published": published}
                )
        except Exception as exc:
            logger.info("RSS fetch failed for query %r: %s", query, exc)
    logger.info("Collected %d unique headlines via Google News RSS", len(ctx.headlines))


def _collect_tavily_headlines(ctx: EngineContext) -> bool:
    keys = os.getenv("TAVILY_API_KEYS") or os.getenv("TAVILY_API_KEY") or ""
    api_key = next((k.strip() for k in keys.split(",") if k.strip()), "")
    if not api_key:
        return False
    try:
        import requests

        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": "global macro economy markets Fed inflation geopolitics",
                "topic": "news",
                "max_results": 12,
                "days": 2,
            },
            timeout=20,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        for r in results:
            title = (r.get("title") or "").strip()
            if not title:
                continue
            snippet = (r.get("content") or "").strip()
            ctx.headlines.append(
                {
                    "title": f"{title} — {snippet[:220]}".strip(" —"),
                    "source": r.get("url", "")[:120],
                    "published": r.get("published_date", "") or "",
                }
            )
        ctx.news_source = "tavily"
        logger.info("Collected %d headlines via Tavily", len(ctx.headlines))
        return bool(ctx.headlines)
    except Exception as exc:
        logger.info("Tavily fetch failed, falling back to RSS: %s", exc)
        return False


def collect_headlines(ctx: EngineContext) -> None:
    if not _collect_tavily_headlines(ctx):
        _collect_rss_headlines(ctx)
    ctx.headlines = ctx.headlines[:24]


# ----------------------------------------------------------------------------
# Step 3: LLM candidate resolution + call
# ----------------------------------------------------------------------------

def _first_env_key(*names: str) -> str:
    for name in names:
        raw = os.getenv(name, "")
        for token in raw.split(","):
            token = token.strip()
            if token:
                return token
    return ""


# Auto-select free Gemini models — no manual choosing needed.
# The engine tries them in order until one works, so it survives individual
# model failures and free-tier changes.
#
# Live, free-tier-eligible models only (curated 2026-09-16):
#   * the 1.5 family was shut down 2025-09-29
#   * the 2.0 family was shut down 2026-06-01
#   * Pro models left the free tier on 2026-04-01 (paid-only)
# A retired id left in this list costs ~9s of pointless retry budget
# (2 attempts + backoff) before the engine reaches a model that answers,
# so keep it pruned — or override with MACRO_SENTIMENT_GEMINI_MODELS.
FREE_GEMINI_MODELS: List[str] = [
    "gemini-3.6-flash",       # GA — Google's documented path off 2.0 Flash
    "gemini-3.5-flash-lite",  # GA — current cheap Flash-Lite tier
    "gemini-3.1-flash-lite",  # GA — long deprecation runway (2027-05-07)
    "gemini-2.5-flash",       # still free; no shutdown date announced
    "gemini-2.5-flash-lite",  # still free; no shutdown date announced
]


def _normalized_gemini_model(raw: str) -> str:
    raw = (raw or "").strip() or FREE_GEMINI_MODELS[0]
    return raw if raw.startswith("gemini/") else f"gemini/{raw}"


def _configured_free_gemini_models() -> List[str]:
    """Free-model auto-select list, overridable without a code change.

    ``MACRO_SENTIMENT_GEMINI_MODELS`` (comma-separated) lets an operator pin the
    models to try when Google retires one of the curated ids. Unset, empty, or
    blank-only values fall back to :data:`FREE_GEMINI_MODELS`.
    """
    raw = (os.getenv("MACRO_SENTIMENT_GEMINI_MODELS") or "").strip()
    if not raw:
        return list(FREE_GEMINI_MODELS)
    models = [part.strip() for part in raw.split(",") if part.strip()]
    return models or list(FREE_GEMINI_MODELS)


def resolve_llm_candidates() -> List[Dict[str, Any]]:
    """Ordered candidate LLMs — first available wins (Gemini primary).

    Auto-select behavior (per user request): if GEMINI_API_KEY is set and
    no explicit GEMINI_MODEL is provided, the system auto-tries all
    FREE_GEMINI_MODELS in order. Manual override via GEMINI_MODEL or
    MACRO_SENTIMENT_MODEL still works for power users.
    """
    candidates: List[Dict[str, Any]] = []

    explicit = (os.getenv("MACRO_SENTIMENT_MODEL") or "").strip()
    if explicit:
        candidates.append({"model": explicit})
        return candidates

    gemini_key = _first_env_key("GEMINI_API_KEYS", "GEMINI_API_KEY")
    if gemini_key:
        explicit_gemini = (os.getenv("GEMINI_MODEL") or "").strip()
        if explicit_gemini:
            # User explicitly chose a model — respect it
            candidates.append({"model": _normalized_gemini_model(explicit_gemini), "api_key": gemini_key})
            fallback_model = (os.getenv("GEMINI_MODEL_FALLBACK") or "").strip()
            if fallback_model:
                candidates.append(
                    {"model": _normalized_gemini_model(fallback_model), "api_key": gemini_key}
                )
        else:
            # Auto-select: try all free models until one works
            for free_model in _configured_free_gemini_models():
                candidates.append(
                    {"model": _normalized_gemini_model(free_model), "api_key": gemini_key}
                )

    aihubmix_key = (os.getenv("AIHUBMIX_KEY") or "").strip()
    openai_key = _first_env_key("OPENAI_API_KEYS", "OPENAI_API_KEY") or aihubmix_key
    if openai_key:
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or (
            "https://aihubmix.com/v1" if aihubmix_key and not os.getenv("OPENAI_API_KEY") else ""
        )
        candidates.append(
            {
                "model": (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip(),
                "api_key": openai_key,
                "base_url": base_url or None,
            }
        )

    anthropic_key = _first_env_key("ANTHROPIC_API_KEYS", "ANTHROPIC_API_KEY")
    if anthropic_key:
        candidates.append(
            {
                "model": (os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022").strip(),
                "api_key": anthropic_key,
            }
        )

    deepseek_key = _first_env_key("DEEPSEEK_API_KEYS", "DEEPSEEK_API_KEY")
    if deepseek_key:
        candidates.append(
            {"model": "deepseek/deepseek-chat", "api_key": deepseek_key}
        )

    litellm_model = (os.getenv("LITELLM_MODEL") or "").strip()
    litellm_key = (os.getenv("LITELLM_API_KEY") or "").strip()
    if litellm_model:
        entry: Dict[str, Any] = {"model": litellm_model}
        if litellm_key:
            entry["api_key"] = litellm_key
        litellm_base = (os.getenv("LITELLM_API_BASE") or "").strip()
        if litellm_base:
            entry["base_url"] = litellm_base
        candidates.append(entry)

    return candidates


def build_prompts(ctx: EngineContext) -> Tuple[str, str]:
    snapshot_block = json.dumps(ctx.snapshot, ensure_ascii=False, indent=1)
    if not snapshot_block or snapshot_block == "[]":
        snapshot_block = "[market data unavailable — rely on headlines]"
    headlines = [
        f"- {h['title']}" + (f" ({h['source']})" if h.get("source") else "")
        for h in ctx.headlines[:20]
    ]
    headlines_block = "\n".join(headlines) or "- No headlines available"
    previous = (
        f"{ctx.previous_score:+.2f}" if ctx.previous_score is not None else "n/a (first run)"
    )
    user_prompt = f"""## Engine run (UTC): {_utc_now_iso()}
## Previous macro_score (for continuity, new data always wins): {previous}

## Snapshot: global market data (1-day change in %)
{snapshot_block}

## Recent macro headlines (last ~48h, English only)
{headlines_block}

Produce the contract JSON now. Remember: {STRICT_ENGLISH_DIRECTIVE}"""
    return SYSTEM_PROMPT, user_prompt


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse an LLM response into a JSON object (tolerant of prose/fences)."""
    cleaned = _strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # Slice from the first '{' to the last '}' then repair.
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        sliced = cleaned[start : end + 1]
        try:
            parsed = json.loads(sliced)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            from json_repair import repair_json

            parsed = repair_json(sliced, return_objects=True)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    raise ValueError("LLM response did not contain a JSON object")


def generate_with_llm(ctx: EngineContext) -> Dict[str, Any]:
    """Try each LLM candidate until one returns a parseable JSON object."""
    candidates = resolve_llm_candidates()
    if not candidates:
        raise RuntimeError("no LLM API key configured")
    system_prompt, user_prompt = build_prompts(ctx)

    import litellm  # lazy heavy import

    try:
        litellm.suppress_debug_info = True
    except Exception:
        pass

    last_error: Optional[BaseException] = None
    for candidate in candidates:
        model = candidate["model"]
        for attempt in range(2):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": float(os.getenv("MACRO_SENTIMENT_TEMPERATURE", "0.2")),
                    "max_tokens": 1600,
                    "timeout": 90,
                }
                if candidate.get("api_key"):
                    kwargs["api_key"] = candidate["api_key"]
                if candidate.get("base_url"):
                    kwargs["api_base"] = candidate["base_url"]
                logger.info(
                    "LLM attempt %s/2 with model %s", attempt + 1, model
                )
                response = litellm.completion(**kwargs)
                content = response.choices[0].message.content or ""
                payload = extract_json_object(content)
                ctx.llm_used = True
                ctx.llm_model = model
                logger.info("LLM %s returned a parseable JSON payload", model)
                return payload
            except Exception as exc:
                last_error = exc
                logger.warning("LLM attempt %s with %s failed: %s", attempt + 1, model, exc)
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"all LLM candidates failed; last error: {last_error}")


# ----------------------------------------------------------------------------
# Step 4: payload coercion (schema safety) + rule-based fallback
# ----------------------------------------------------------------------------

def _coerce_str_list(value: Any, limit: int = 12, drop_cjk: bool = True) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        if drop_cjk and _contains_cjk(text):
            # Strict-English contract: CJK content can never reach TeacherBrain
            # / Hermes (pattern matching is English-token based).
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _sentiment_from_score(score: float) -> str:
    if score >= 0.15:
        return "bullish"
    if score <= -0.15:
        return "bearish"
    return "neutral"


def _coerce_score_from_sentiment(sentiment: str) -> float:
    return {"bullish": 0.5, "bearish": -0.5}.get(sentiment, 0.0)


def _fallback_summary(snapshot: List[Dict[str, Any]]) -> str:
    vix = next((s for s in snapshot if s.get("role") == "volatility"), None)
    equities = [s for s in snapshot if s.get("role") == "equity"]
    us_equities = [s for s in equities if str(s.get("symbol", "")).startswith("^")]
    advancers = sum(
        1 for s in equities if (_to_float(s.get("change_pct_1d")) or 0) > 0
    )
    parts = []
    if us_equities:
        spx = us_equities[0]
        spx_name = str(spx.get("name") or "US indices")
        spx_move = _to_float(spx.get("change_pct_1d"))
        if spx_move is None:
            parts.append(
                f"US equities closed a mixed session ({spx_name}, latest "
                f"{spx.get('last')}) with {advancers} of {len(equities)} tracked "
                "global indices advancing."
            )
        else:
            direction = "higher" if spx_move > 0 else "lower"
            parts.append(
                f"US equities traded {direction} ({spx_name} "
                f"{spx_move:+.2f}% on the session) with {advancers} of "
                f"{len(equities)} tracked global indices advancing."
            )
    elif equities:
        parts.append(
            f"{advancers} of {len(equities)} tracked global indices advanced on the session."
        )
    vix_level = _to_float(vix.get("last")) if vix else None
    if vix_level is not None:
        parts.append(
            f"Volatility (VIX {vix_level:.1f}) implies "
            + (
                "an elevated risk regime; stops and position sizing should stay defensive."
                if vix_level >= 25
                else "a contained risk regime, with no systemic stress visible in market data."
            )
        )
    else:
        parts.append(
            "Market data was partially unavailable; this assessment is rule-based and "
            "should be treated with reduced confidence."
        )
    return " ".join(parts[:2])


def _rule_based_payload(ctx: EngineContext) -> Dict[str, Any]:
    """Deterministic macro assessment from market data alone (no LLM).

    Defensive by construction: every value from `ctx` is sanitized through
    `_to_float` / `.get()` so this safety net can never crash the run.
    """
    score = 0.0
    warnings: List[str] = []
    drivers: List[str] = []

    vix = next((s for s in ctx.snapshot if s.get("role") == "volatility"), None)
    vix_level = _to_float(vix.get("last")) if vix else None
    if vix_level is not None:
        if vix_level >= 32:
            score -= 0.55
            warnings.append(
                f"VIX at {vix_level:.1f} (panic regime) — widen ATR stops / consider halting trading"
            )
        elif vix_level >= 25:
            score -= 0.35
            warnings.append(f"VIX at {vix_level:.1f} — elevated volatility regime")
        elif vix_level >= 20:
            score -= 0.1
        else:
            score += 0.2
        chg = _to_float(vix.get("change_pct_1d"))
        if chg is not None and chg >= 10:
            score -= 0.15
            warnings.append(f"VIX spiked {chg:+.1f}% in one session")
        drivers.append(f"VIX {vix_level:.1f}")

    equity_moves = [
        (str(s.get("name") or s.get("symbol") or "index"), move)
        for s in ctx.snapshot
        if s.get("role") == "equity"
        for move in [_to_float(s.get("change_pct_1d"))]
        if move is not None
    ]
    if equity_moves:
        avg_move = sum(m for _, m in equity_moves) / len(equity_moves)
        score += _clamp(avg_move / 3.0) * 0.5
        for name, move in equity_moves:
            if move <= -2.5:
                warnings.append(f"{name} {move:+.2f}% single-session rout")
            if move <= -3.0:
                score -= 0.15
        drivers.append(f"Global equities {avg_move:+.2f}% avg")

    ten_y = next((s for s in ctx.snapshot if s.get("role") == "rates"), None)
    ten_y_level = _to_float(ten_y.get("last")) if ten_y else None
    if ten_y_level is not None:
        if ten_y_level >= 4.5:
            score -= 0.15
            warnings.append(f"US 10Y yield at {ten_y_level:.2f}% — tightening headwind")
        elif ten_y_level <= 3.5:
            score += 0.05
        drivers.append(f"US 10Y {ten_y_level:.2f}%")

    dxy = next((s for s in ctx.snapshot if s.get("role") == "fx"), None)
    dxy_move = _to_float(dxy.get("change_pct_1d")) if dxy else None
    if dxy_move is not None:
        if dxy_move >= 0.5:
            score -= 0.1
            drivers.append("USD surge (risk-off)")
        elif dxy_move <= -0.5:
            score += 0.1
            drivers.append("USD relief rally")

    for spec in ctx.snapshot:
        move = _to_float(spec.get("change_pct_1d"))
        if move is None:
            continue
        name = str(spec.get("name") or spec.get("symbol") or "instrument")
        if spec.get("role") == "commodity" and "Oil" in name and move >= 3.0:
            score -= 0.1
            drivers.append(f"{name} +{move:.1f}% oil spike")
        if spec.get("role") == "crypto" and move <= -5.0:
            score -= 0.1
            warnings.append(f"Bitcoin {move:+.1f}% crash — crypto risk contagion watch")

    # Headline keyword scan feeds Hermes black-swan awareness (English only).
    keyword_hits = 0
    for h in ctx.headlines[:24]:
        title = str(h.get("title") or "").strip()
        if not title or _contains_cjk(title):
            continue
        low = title.lower()
        for kw in _HEADLINE_RISK_KEYWORDS:
            if kw in low:
                warnings.append(f"Headline risk: {title}")
                keyword_hits += 1
                break
        if keyword_hits >= 4:
            break
    if keyword_hits:
        score -= min(0.15, 0.05 * keyword_hits)

    # Surface top catalysts from headline titles (stable English tokens).
    for h in ctx.headlines[:6]:
        title = str(h.get("title") or "").strip()
        if title and not _contains_cjk(title):
            drivers.append(title.split(" - ")[0][:80])

    score = round(_clamp(score), 2)
    payload = {
        "market_sentiment": _sentiment_from_score(score),
        "macro_score": score,
        "hermes_risk_feed": {
            "summary": _fallback_summary(ctx.snapshot),
            "black_swan_warnings": _dedupe(warnings)[:8],
        },
        "teacher_brain_catalysts": {"key_drivers": _dedupe(drivers)[:8]},
    }
    return payload


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def coerce_payload(raw: Dict[str, Any], ctx: EngineContext) -> Dict[str, Any]:
    """Coerce an LLM (or fallback) payload into the exact output contract."""
    if not isinstance(raw, dict):
        raw = {}

    score_raw = raw.get("macro_score")
    try:
        if isinstance(score_raw, bool):
            raise TypeError("bool is not a valid macro_score")
        score = _clamp(round(float(score_raw), 2))
    except (TypeError, ValueError):
        score = None

    sentiment_raw = str(raw.get("market_sentiment") or "").strip().lower()
    if sentiment_raw not in ("bullish", "bearish", "neutral"):
        sentiment_raw = ""

    if score is None:
        score = _coerce_score_from_sentiment(sentiment_raw or "neutral")
    if not sentiment_raw:
        sentiment_raw = _sentiment_from_score(score)

    hermes = raw.get("hermes_risk_feed") if isinstance(raw.get("hermes_risk_feed"), dict) else {}
    summary = str(hermes.get("summary") or "").strip()
    if not summary or _contains_cjk(summary):
        summary = _fallback_summary(ctx.snapshot)
    warnings = _coerce_str_list(hermes.get("black_swan_warnings"), limit=8)

    teacher = (
        raw.get("teacher_brain_catalysts")
        if isinstance(raw.get("teacher_brain_catalysts"), dict)
        else {}
    )
    drivers = _coerce_str_list(teacher.get("key_drivers"), limit=8)
    if not drivers:
        drivers = _coerce_str_list(
            [
                title
                for title in (str(h.get("title") or "") for h in ctx.headlines[:5])
                if not _contains_cjk(title)
            ],
            limit=5,
        )

    return {
        "market_sentiment": sentiment_raw,
        "macro_score": score,
        "timestamp_utc": _utc_now_iso(),
        "hermes_risk_feed": {
            "summary": summary,
            "black_swan_warnings": _dedupe(warnings)[:8],
        },
        "teacher_brain_catalysts": {"key_drivers": _dedupe(drivers)[:8]},
    }


# ----------------------------------------------------------------------------
# Step 4.5: runtime contract validation + minimal last-resort payload
# ----------------------------------------------------------------------------

_TOP_LEVEL_SCHEMA: Dict[str, Any] = {
    "market_sentiment": str,
    "macro_score": (int, float),
    "timestamp_utc": str,
    "hermes_risk_feed": dict,
    "teacher_brain_catalysts": dict,
}


def _validate_contract(payload: Dict[str, Any]) -> None:
    """Strict runtime check of the Genie Trader Pro JSON contract.

    Raises ValueError on ANY deviation (missing/extra key, wrong type, score
    out of range, invalid sentiment, non-list string collections, CJK text).
    """
    if not isinstance(payload, dict):
        raise ValueError("payload is not a dict")
    if set(payload.keys()) != set(_TOP_LEVEL_SCHEMA.keys()):
        raise ValueError(f"top-level keys mismatch: {sorted(payload.keys())}")
    for key, expected in _TOP_LEVEL_SCHEMA.items():
        if not isinstance(payload.get(key), expected):
            raise ValueError(f"key {key!r} has wrong type: {type(payload.get(key))}")
    if payload["market_sentiment"] not in ("bullish", "bearish", "neutral"):
        raise ValueError("market_sentiment out of enum")
    score = payload["macro_score"]
    if not -1.0 <= float(score) <= 1.0:
        raise ValueError("macro_score out of [-1.0, 1.0]")
    if not (isinstance(payload["timestamp_utc"], str) and payload["timestamp_utc"].endswith("Z")):
        raise ValueError("timestamp_utc is not ISO-8601 UTC")
    hermes = payload["hermes_risk_feed"]
    if set(hermes.keys()) != {"summary", "black_swan_warnings"}:
        raise ValueError("hermes_risk_feed keys mismatch")
    if not isinstance(hermes["summary"], str) or not hermes["summary"]:
        raise ValueError("summary must be a non-empty string")
    if not isinstance(hermes["black_swan_warnings"], list) or not all(
        isinstance(w, str) for w in hermes["black_swan_warnings"]
    ):
        raise ValueError("black_swan_warnings must be a list of strings")
    teacher = payload["teacher_brain_catalysts"]
    if set(teacher.keys()) != {"key_drivers"}:
        raise ValueError("teacher_brain_catalysts keys mismatch")
    if not isinstance(teacher["key_drivers"], list) or not all(
        isinstance(d, str) for d in teacher["key_drivers"]
    ):
        raise ValueError("key_drivers must be a list of strings")
    serialized = json.dumps(payload, ensure_ascii=False)
    if _contains_cjk(serialized):
        raise ValueError("payload contains CJK characters (strict-English contract)")


def _minimal_contract_payload() -> Dict[str, Any]:
    """Last-resort, always-valid payload (used only if everything else fails)."""
    return {
        "market_sentiment": "neutral",
        "macro_score": 0.0,
        "timestamp_utc": _utc_now_iso(),
        "hermes_risk_feed": {
            "summary": (
                "Sentiment data unavailable for this cycle; a minimal neutral "
                "assessment was emitted, so treat signals with reduced confidence."
            ),
            "black_swan_warnings": [],
        },
        "teacher_brain_catalysts": {"key_drivers": []},
    }


# ----------------------------------------------------------------------------
# Step 5: atomic write
# ----------------------------------------------------------------------------

def write_payload(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=output_path.name, dir=str(output_path.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(serialized)
        os.replace(tmp_name, output_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def run(output_path: Optional[str] = None) -> int:
    """Execute one engine cycle; always leaves a valid JSON file behind.

    Returns 0 on success (JSON written — even via fallback), non-zero only if
    the file itself cannot be produced.
    """
    start = time.time()
    target = resolve_output_path(output_path)
    logger.info("=" * 60)
    logger.info("Macro-Sentiment & Risk Engine v%s starting", ENGINE_VERSION)
    logger.info("Output target: %s", target)

    ctx = EngineContext()
    ctx.previous_score = load_previous_score(target)

    collect_market_snapshot(ctx)
    collect_headlines(ctx)

    raw_payload: Optional[Dict[str, Any]] = None
    try:
        raw_payload = generate_with_llm(ctx)
    except Exception as exc:
        ctx.fallback_reason = str(exc)
        logger.warning("LLM generation unavailable, using rule-based engine: %s", exc)

    if raw_payload is None:
        try:
            raw_payload = _rule_based_payload(ctx)
        except Exception:
            # The fallback itself must never kill the run (audit hardening).
            logger.exception("Rule-based engine failed; degrading to minimal contract")
            raw_payload = None

    try:
        payload = coerce_payload(raw_payload or {}, ctx)
        _validate_contract(payload)
    except Exception:
        logger.exception("Payload failed contract coercion/validation; using minimal contract")
        payload = _minimal_contract_payload()
        # The minimal contract is static and valid; validate as a paranoia gate.
        _validate_contract(payload)

    write_payload(payload, target)

    mode = f"llm:{ctx.llm_model}" if ctx.llm_used else f"fallback:{ctx.fallback_reason or 'n/a'}"
    duration = time.time() - start
    logger.info(
        "sentiment=%s macro_score=%+.2f warnings=%d drivers=%d (%s, %.1fs)",
        payload["market_sentiment"],
        payload["macro_score"],
        len(payload["hermes_risk_feed"]["black_swan_warnings"]),
        len(payload["teacher_brain_catalysts"]["key_drivers"]),
        mode,
        duration,
    )
    # Machine-parseable line for CI logs / workflow debugging.
    print(
        "SENTIMENT_JSON_WRITTEN "
        + json.dumps(
            {
                "path": str(target),
                "market_sentiment": payload["market_sentiment"],
                "macro_score": payload["macro_score"],
                "mode": mode,
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    argv = list(argv if argv is not None else sys.argv[1:])
    output = argv[0] if argv else None
    return run(output_path=output)


if __name__ == "__main__":
    raise SystemExit(main())
