# -*- coding: utf-8 -*-
"""Single source of truth for the auto-selected (free-tier) Gemini model list.

This module is intentionally dependency-free (a leaf import) so it can be
shared by both the main app config (:mod:`src.config`) and the standalone
Macro-Sentiment & Risk Engine (:mod:`src.macro_sentiment.engine`) without any
import cycle and without touching the network at import time.

Curated 2026-09-16 — live, free-tier-eligible models only:

  * the 1.5 family was shut down 2025-09-29
  * the 2.0 family was shut down 2026-06-01
  * Pro models left the free tier on 2026-04-01 (paid-only), so
    ``gemini-3.1-pro-preview`` must never come back as a default

A retired id left in this list costs ~9s of pointless retry budget (2 attempts
+ backoff) before a model that answers is reached, so keep it pruned.

Overrides (explicit config always wins over auto-select):
  * main app:   ``GEMINI_MODEL`` / ``GEMINI_MODEL_FALLBACK`` / ``LITELLM_MODEL``
  * macro feed: ``MACRO_SENTIMENT_MODEL`` / ``GEMINI_MODEL`` /
    ``MACRO_SENTIMENT_GEMINI_MODELS``

``tests/test_model_selection.py`` guards the drift between this list, the
defaults derived from it, and the retired-model registry.
"""

from __future__ import annotations

from typing import List

# Ordered: the first entry is the primary, the rest are tried in order.
FREE_GEMINI_MODELS: List[str] = [
    "gemini-3.6-flash",       # GA — Google's documented path off 2.0 Flash
    "gemini-3.5-flash-lite",  # GA — current cheap Flash-Lite tier
    "gemini-3.1-flash-lite",  # GA — long deprecation runway (2027-05-07)
    "gemini-2.5-flash",       # still free; no shutdown date announced
    "gemini-2.5-flash-lite",  # still free; no shutdown date announced
]

# Defaults derived from FREE_GEMINI_MODELS — never hardcode a model name at a
# call site, so a single edit here migrates every entry point at once.
DEFAULT_GEMINI_MODEL: str = FREE_GEMINI_MODELS[0]
DEFAULT_GEMINI_MODEL_FALLBACK: str = FREE_GEMINI_MODELS[1]

# Router fallback depth for the auto-selected chain. Longer chains only add
# retry budget without improving the odds of a usable answer.
AUTO_SELECT_MAX_FALLBACKS: int = 2


def free_gemini_primary_model() -> str:
    """Bare primary model id (no ``gemini/`` provider prefix)."""
    return DEFAULT_GEMINI_MODEL


def free_gemini_fallback_models(exclude: str = "") -> List[str]:
    """Ordered bare fallback ids from the free-tier list.

    ``exclude`` drops the model already used as the primary. Blank entries are
    ignored, the result is deduped, order-preserving, and capped at
    :data:`AUTO_SELECT_MAX_FALLBACKS`.
    """
    skip = (exclude or "").strip()
    out: List[str] = []
    for model in [DEFAULT_GEMINI_MODEL_FALLBACK, *FREE_GEMINI_MODELS]:
        model = (model or "").strip()
        if not model or model == skip or model in out:
            continue
        out.append(model)
        if len(out) >= AUTO_SELECT_MAX_FALLBACKS:
            break
    return out


__all__ = [
    "AUTO_SELECT_MAX_FALLBACKS",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GEMINI_MODEL_FALLBACK",
    "FREE_GEMINI_MODELS",
    "free_gemini_fallback_models",
    "free_gemini_primary_model",
]
