# Macro-Sentiment & Risk Engine (Genie Trader Pro Feed)

This repository runs as a **fully hands-off macro-sentiment producer** for the
Genie Trader Pro algorithmic trading engine. A GitHub Actions workflow runs on
a 4-hour cron, generates one strict-English JSON file —
**`sentiment_latest.json`** — and commits it back to `main`. Genie Trader Pro
polls the file over `raw.githubusercontent.com`; no human touches the pipeline
after the one-time secrets setup.

## Architecture

```
┌─────────────────────────── GitHub Actions (cron: every 4h) ───────────────────────────┐
│  yfinance snapshot ──┐                                                                │
│  (indices/VIX/DXY/   │                                                                │
│   10Y/oil/gold/BTC)  │   ┌──────────────────────┐    ┌─────────────────────────────┐  │
│                      ├──▶│ LLM (Gemini default, │───▶│  schema coercion + clamping │──┼──▶ sentiment_latest.json
│  macro headlines ────┤   │  strict-English JSON │    │  (CJK scrub, score ∈ [-1,1])│  │    (committed to main)
│  (Tavily or free RSS)│   └─────────┬────────────┘    └─────────────────────────────┘  │
│                      │             │ on any failure                                    │
│                      └─────────────┴──────────────────────────▶ rule-based fallback    │
│                                   (deterministic scoring: always emits valid JSON)   │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

- **Entry point:** `python main.py --macro-sentiment`
  (standalone equivalent: `python -m src.macro_sentiment.engine`)
- **Engine package:** `src/macro_sentiment/engine.py`
- **Workflow:** `.github/workflows/macro-sentiment-engine.yml`
- **Tests:** `tests/test_macro_sentiment_engine.py`

The mode bypasses the markdown report pipeline and notifications entirely —
its only side effect is the JSON file (written atomically via tmp + rename).

## Output contract (`sentiment_latest.json`)

Exactly five top-level keys, in this order, always valid JSON:

```json
{
  "market_sentiment": "bullish | bearish | neutral",
  "macro_score": 0.85,
  "timestamp_utc": "2026-09-14T10:07:00Z",
  "hermes_risk_feed": {
    "summary": "Exactly two English sentences describing the macro state.",
    "black_swan_warnings": [
      "High-impact risks intended to trigger Hermes ATR-stop widening / trading halts"
    ]
  },
  "teacher_brain_catalysts": {
    "key_drivers": [
      "Short stable English catalyst tokens for TeacherBrain SQLite pattern matching"
    ]
  }
}
```

Guarantees enforced by code (`coerce_payload`), not just by the prompt:

| Field | Guarantee |
| --- | --- |
| `macro_score` | float, clamped to `[-1.0, 1.0]`, two decimals |
| `market_sentiment` | derived from score when missing/invalid (`>=0.15` bullish, `<=-0.15` bearish) |
| `timestamp_utc` | ISO-8601 UTC, set by the engine (never by the LLM) |
| all human-readable strings | CJK scrubbed — Chinese text can never reach TeacherBrain/Hermes |

### Consuming side examples

```python
# Genie Trader Pro — poll the feed (public repo needs no auth; private repos
# need a fine-grained read-only token in the Authorization header).
import requests

FEED_URL = (
    "https://raw.githubusercontent.com/himanshuverma993/"
    "daily_stock_analysis/main/sentiment_latest.json"
)
feed = requests.get(FEED_URL, timeout=15).json()

# HermesBrain — risk overrides
if feed["macro_score"] <= -0.5 or feed["hermes_risk_feed"]["black_swan_warnings"]:
    hermes.widen_atr_stops(factor=1.5)          # or hermes.halt_trading()

# TeacherBrain — catalyst pattern matching
for catalyst in feed["teacher_brain_catalysts"]["key_drivers"]:
    teacher.record_catalyst(catalyst, score=feed["macro_score"])
```

## Workflow behavior

- **Cron:** `7 */4 * * *` (every 4 hours, minute 7 — off the congested top-of-hour mark), plus `workflow_dispatch` for manual runs (with an optional one-off model override input).
- **Permissions:** `contents: write`, commit author is `github-actions[bot]`; commit message is `chore(sentiment): refresh sentiment_latest.json [skip ci]` (`[skip ci]` avoids CI loops).
- **Idempotency:** commits only when `sentiment_latest.json` changes.
- **Rebase-safe:** `git pull --rebase --autostash` before push, so overlapping daily-JSON commits never fail the push.
- **Resilience (defense in depth):** any LLM/data failure falls back to a deterministic rule-based payload (which is itself defensively coded — `_to_float` everywhere, partial market data tolerated); the coerced result must then pass a strict runtime `validate_contract` gate (exact keys, types, score bounds, non-empty summary, list-of-strings collections, no CJK); if even that is somehow violated, a static minimal contract payload is written instead. The workflow turns red only if a valid JSON file physically cannot be written.
- **Artifact:** each run also uploads the JSON as a build artifact (30-day retention audit trail).

## Secrets report — exact keys to configure

Repository path: **Settings → Secrets and variables → Actions → New repository secret**.

### Required (exactly ONE of these — Gemini recommended)

| Secret | Purpose | Get it at |
| --- | --- | --- |
| `GEMINI_API_KEY` | Primary LLM (auto-selects free models: 2.0-flash, 1.5-flash, 2.5-flash etc.) | https://aistudio.google.com/apikey (free tier) |
| `OPENAI_API_KEY` | Fallback LLM #1 | https://platform.openai.com/api-keys |
| `AIHUBMIX_KEY` | OpenAI-compatible aggregator (engine auto-uses `https://aihubmix.com/v1`) | https://aihubmix.com |
| `ANTHROPIC_API_KEY` | Fallback LLM #2 (engine default `claude-3-5-sonnet-20241022`) | https://console.anthropic.com |
| `DEEPSEEK_API_KEY` | Fallback LLM #3 (`deepseek/deepseek-chat`) | https://platform.deepseek.com |

If **none** is set, the engine still runs on the rule-based fallback — but for
real signal quality, configure at least one.

> **Auto-select free Gemini models:** When only `GEMINI_API_KEY` is set (no `GEMINI_MODEL`),
> the engine auto-tries `gemini-2.0-flash` → `2.0-flash-lite` → `1.5-flash` → `1.5-flash-8b`
> → `2.5-flash` → `2.5-flash-lite` → `1.5-pro` until one works. No manual model
> choosing needed — survives deprecations and free-tier changes. Power users can
> still force a model via `GEMINI_MODEL` or `MACRO_SENTIMENT_MODEL`.

### Optional

| Secret / Variable | Purpose | Default if unset |
| --- | --- | --- |
| `TAVILY_API_KEYS` (secret) | Richer macro headline search | Free Google News RSS (no key) |
| `GEMINI_MODEL` (variable) | Force a specific Gemini model (optional, auto-selects if empty) | auto-select free list |
| `GEMINI_MODEL_FALLBACK` (variable) | Second choice when GEMINI_MODEL is forced | none (auto list used when GEMINI_MODEL empty) |
| `OPENAI_MODEL` (variable) | OpenAI model | `gpt-4o-mini` |
| `OPENAI_BASE_URL` (variable) | Custom OpenAI-compatible endpoint | official API |
| `ANTHROPIC_MODEL` (variable) | Claude model | `claude-3-5-sonnet-20241022` |
| `LITELLM_MODEL` + `LITELLM_API_KEY` (secret/variable) | Any other litellm-supported provider (e.g. `openrouter/...`) | off |
| `MACRO_SENTIMENT_MODEL` (variable) | Force a litellm model id | auto priority above |
| `MACRO_SENTIMENT_OUTPUT` (variable) | Custom output path | `sentiment_latest.json` |
| `MACRO_SENTIMENT_TEMPERATURE` (variable) | LLM temperature | `0.2` |

### Repository setting (one click, not a secret)

**Settings → Actions → General → Workflow permissions → "Read and write permissions"**
— required so the Actions bot can push `sentiment_latest.json` back to `main`.

## Local execution

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...        # or any other provider key
python main.py --macro-sentiment
# inspect
cat sentiment_latest.json
```

Deterministic no-key sanity check:

```bash
python -m src.macro_sentiment.engine /tmp/sentiment.json   # uses rule-based fallback
```

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `mode: fallback:no LLM API key configured` in logs | No recognized LLM secret — add `GEMINI_API_KEY` |
| Workflow fails at push step | Workflow permissions not set to "Read and write" |
| All market instruments fail | Transient Yahoo throttling; engine degrades gracefully, next run recovers |
| JSON contains Chinese | Impossible by design — `coerce_payload` scrubs CJK from every human-readable value |
