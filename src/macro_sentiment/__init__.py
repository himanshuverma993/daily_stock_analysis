# -*- coding: utf-8 -*-
"""Macro-Sentiment & Risk Engine (Genie Trader Pro feed).

This package turns the repository into a hands-off macro-sentiment producer:
it aggregates global market data and macro headlines, asks an LLM for a
strictly-English risk assessment (with a deterministic rule-based fallback),
and persists a single machine-readable contract file:

    sentiment_latest.json

The JSON contract is designed for the Genie Trader Pro autonomous learning
modules:

- ``HermesBrain``  -> ``hermes_risk_feed`` (black-swan warnings that should
  trigger ATR-stop widening or trading halts).
- ``TeacherBrain`` -> ``teacher_brain_catalysts`` (stable catalyst tokens for
  SQLite pattern matching).
"""

from src.macro_sentiment.engine import (
    ENGINE_VERSION,
    OUTPUT_FILENAME,
    run,
)

__all__ = ["ENGINE_VERSION", "OUTPUT_FILENAME", "run"]
