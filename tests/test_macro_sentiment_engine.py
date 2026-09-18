# -*- coding: utf-8 -*-
"""Tests for the Macro-Sentiment & Risk Engine (Genie Trader Pro feed)."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.macro_sentiment import engine


CONTRACT_KEYS = [
    "market_sentiment",
    "macro_score",
    "timestamp_utc",
    "hermes_risk_feed",
    "teacher_brain_catalysts",
]

# Gemini ids Google has already shut down (1.5 family 2025-09-29, 2.0 family
# 2026-06-01, plus their dated snapshots). Guards against re-adding dead models.
RETIRED_GEMINI_MODELS = {
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
}


def _snapshot_vix(level: float, change: float = 0.0):
    return [
        {"symbol": "^GSPC", "name": "S&P 500", "role": "equity", "last": 6500.0, "change_pct_1d": 0.4},
        {"symbol": "^VIX", "name": "CBOE VIX", "role": "volatility", "last": level, "change_pct_1d": change},
        {"symbol": "^TNX", "name": "US 10Y Treasury Yield", "role": "rates", "last": 4.2, "change_pct_1d": 0.0},
    ]


class ContractTestCase(unittest.TestCase):
    """The fallback path must always emit the exact Genie Trader Pro contract."""

    def test_empty_inputs_still_produce_valid_payload(self):
        ctx = engine.EngineContext()
        payload = engine.coerce_payload(engine._rule_based_payload(ctx), ctx)
        self.assertEqual(list(payload.keys()), CONTRACT_KEYS)
        self.assertIn(payload["market_sentiment"], ("bullish", "bearish", "neutral"))
        self.assertGreaterEqual(payload["macro_score"], -1.0)
        self.assertLessEqual(payload["macro_score"], 1.0)
        self.assertIsInstance(payload["timestamp_utc"], str)
        self.assertTrue(payload["timestamp_utc"].endswith("Z"))
        self.assertIsInstance(payload["hermes_risk_feed"]["summary"], str)
        self.assertIsInstance(payload["hermes_risk_feed"]["black_swan_warnings"], list)
        self.assertIsInstance(payload["teacher_brain_catalysts"]["key_drivers"], list)

    def test_no_ascii_order_flip_between_sentiment_and_score(self):
        for score, expected in ((0.85, "bullish"), (-0.85, "bearish"), (0.0, "neutral")):
            payload = engine.coerce_payload({"macro_score": score}, engine.EngineContext())
            self.assertEqual(payload["market_sentiment"], expected)
            self.assertEqual(payload["macro_score"], score)

    def test_score_clamped_and_sentiment_derived(self):
        payload = engine.coerce_payload({"macro_score": 7.5}, engine.EngineContext())
        self.assertEqual(payload["macro_score"], 1.0)
        self.assertEqual(payload["market_sentiment"], "bullish")

    def test_cjk_values_never_reach_the_contract(self):
        raw = {
            "market_sentiment": "bearish",
            "macro_score": -0.6,
            "hermes_risk_feed": {
                "summary": "市场正在经历恐慌性抛售，需要停止交易。",
                "black_swan_warnings": ["美联储紧急加息", "CPI shock"],
            },
            "teacher_brain_catalysts": {"key_drivers": ["加息预期", "OPEC supply cut"]},
        }
        payload = engine.coerce_payload(raw, engine.EngineContext())
        self.assertFalse(engine._contains_cjk(payload["hermes_risk_feed"]["summary"]))
        warnings = payload["hermes_risk_feed"]["black_swan_warnings"]
        self.assertEqual(warnings, ["CPI shock"])
        drivers = payload["teacher_brain_catalysts"]["key_drivers"]
        self.assertEqual(drivers, ["OPEC supply cut"])


class FallbackScoringTestCase(unittest.TestCase):
    def test_panic_vix_yields_bearish_payload_with_warnings(self):
        ctx = engine.EngineContext(snapshot=_snapshot_vix(38.0, change=12.0))
        payload = engine.coerce_payload(engine._rule_based_payload(ctx), ctx)
        self.assertEqual(payload["market_sentiment"], "bearish")
        self.assertLess(payload["macro_score"], -0.15)
        joined = " ".join(payload["hermes_risk_feed"]["black_swan_warnings"])
        self.assertIn("VIX", joined)

    def test_calm_market_yields_non_bearish_payload(self):
        ctx = engine.EngineContext(snapshot=_snapshot_vix(13.5))
        payload = engine.coerce_payload(engine._rule_based_payload(ctx), ctx)
        self.assertNotEqual(payload["market_sentiment"], "bearish")
        self.assertGreaterEqual(payload["macro_score"], 0.0)

    def test_headline_keywords_surface_risk(self):
        ctx = engine.EngineContext(
            snapshot=_snapshot_vix(19.0),
            headlines=[
                {"title": "Major bank faces collapse risk amid contagion fears", "source": "Reuters", "published": ""},
                {"title": "Markets steady ahead of CPI", "source": "AP", "published": ""},
            ],
        )
        payload = engine.coerce_payload(engine._rule_based_payload(ctx), ctx)
        warnings = payload["hermes_risk_feed"]["black_swan_warnings"]
        self.assertTrue(any("Headline risk" in w for w in warnings))


class JsonExtractionTestCase(unittest.TestCase):
    def test_plain_json(self):
        obj = engine.extract_json_object('{"macro_score": 0.2, "market_sentiment": "bullish"}')
        self.assertEqual(obj["macro_score"], 0.2)

    def test_fenced_json(self):
        text = "```json\n{\"macro_score\": -0.4}\n```"
        self.assertEqual(engine.extract_json_object(text)["macro_score"], -0.4)

    def test_prose_wrapped_json(self):
        text = "Here is my analysis:\n{\"market_sentiment\": \"neutral\", \"macro_score\": 0.0}\nDone."
        self.assertEqual(engine.extract_json_object(text)["market_sentiment"], "neutral")

    def test_broken_json_repaired(self):
        text = '{"market_sentiment": "bullish", "macro_score": 0.9,}'
        self.assertEqual(engine.extract_json_object(text)["macro_score"], 0.9)

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            engine.extract_json_object("no json here")


class LlmCandidateResolutionTestCase(unittest.TestCase):
    def test_priority_gemini_first(self):
        env = {
            "GEMINI_API_KEY": "gk",
            "OPENAI_API_KEY": "ok",
            "ANTHROPIC_API_KEY": "ak",
            "MACRO_SENTIMENT_MODEL": "",
            "GEMINI_MODEL": "gemini-2.5-flash",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            candidates = engine.resolve_llm_candidates()
        self.assertEqual(candidates[0]["model"], "gemini/gemini-2.5-flash")
        self.assertEqual(candidates[0]["api_key"], "gk")
        self.assertEqual(candidates[1]["model"], "gpt-4o-mini")
        self.assertIn("claude", candidates[2]["model"])

    def test_auto_select_free_gemini_models(self):
        # When no GEMINI_MODEL is set, engine auto-tries free models list
        env = {
            "GEMINI_API_KEY": "gk",
            "MACRO_SENTIMENT_MODEL": "",
            "GEMINI_MODEL": "",
            "GEMINI_MODEL_FALLBACK": "",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            candidates = engine.resolve_llm_candidates()
        # Should have FREE_GEMINI_MODELS count as Gemini candidates
        self.assertGreaterEqual(len(candidates), len(engine.FREE_GEMINI_MODELS))
        self.assertEqual(candidates[0]["model"], f"gemini/{engine.FREE_GEMINI_MODELS[0]}")
        self.assertEqual(candidates[1]["model"], f"gemini/{engine.FREE_GEMINI_MODELS[1]}")
        # All should have same API key
        for c in candidates[: len(engine.FREE_GEMINI_MODELS)]:
            self.assertEqual(c["api_key"], "gk")

    def test_auto_select_list_has_no_retired_models(self):
        # A retired id in the auto-select list costs ~9s of retry budget per
        # attempt cycle before a live model is reached — keep the list pruned.
        overlap = RETIRED_GEMINI_MODELS & set(engine.FREE_GEMINI_MODELS)
        self.assertEqual(
            overlap, set(), f"retired Gemini models in auto-select list: {sorted(overlap)}"
        )

    def test_blank_model_normalizes_to_live_default(self):
        default = engine._normalized_gemini_model("")
        self.assertEqual(default, f"gemini/{engine.FREE_GEMINI_MODELS[0]}")
        self.assertNotIn(engine.FREE_GEMINI_MODELS[0], RETIRED_GEMINI_MODELS)

    def test_auto_select_models_env_override(self):
        env = {
            "GEMINI_API_KEY": "gk",
            "MACRO_SENTIMENT_MODEL": "",
            "GEMINI_MODEL": "",
            "GEMINI_MODEL_FALLBACK": "",
            "MACRO_SENTIMENT_GEMINI_MODELS": "gemini-3.6-flash, gemini-2.5-flash",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            candidates = engine.resolve_llm_candidates()
        self.assertEqual(
            [c["model"] for c in candidates[:2]],
            ["gemini/gemini-3.6-flash", "gemini/gemini-2.5-flash"],
        )

    def test_blank_env_override_falls_back_to_curated_list(self):
        env = {
            "GEMINI_API_KEY": "gk",
            "MACRO_SENTIMENT_MODEL": "",
            "GEMINI_MODEL": "",
            "GEMINI_MODEL_FALLBACK": "",
            "MACRO_SENTIMENT_GEMINI_MODELS": "  ,  ",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            candidates = engine.resolve_llm_candidates()
        self.assertEqual(candidates[0]["model"], f"gemini/{engine.FREE_GEMINI_MODELS[0]}")
        self.assertEqual(
            len(candidates[: len(engine.FREE_GEMINI_MODELS)]), len(engine.FREE_GEMINI_MODELS)
        )

    def test_explicit_model_override_wins(self):
        with mock.patch.dict(os.environ, {"MACRO_SENTIMENT_MODEL": "openrouter/foo/bar", "GEMINI_API_KEY": "gk"}, clear=True):
            candidates = engine.resolve_llm_candidates()
        self.assertEqual(candidates, [{"model": "openrouter/foo/bar"}])

    def test_aihubmix_base_url_convention(self):
        with mock.patch.dict(os.environ, {"AIHUBMIX_KEY": "hub", "OPENAI_BASE_URL": ""}, clear=True):
            candidates = engine.resolve_llm_candidates()
        self.assertEqual(candidates[0]["base_url"], "https://aihubmix.com/v1")


class DriftGuardTestCase(unittest.TestCase):
    """Drift-guard tests ensuring consistency across engine and main app defaults."""

    def test_auto_select_list_and_config_registry_defaults_match(self):
        """Free-tier auto-select candidates must be live and free-tier eligible."""
        self.assertNotIn(engine.FREE_GEMINI_MODELS[0], RETIRED_GEMINI_MODELS)
        self.assertTrue(all(m not in RETIRED_GEMINI_MODELS for m in engine.FREE_GEMINI_MODELS))

    def test_auto_select_list_is_ordered_and_non_empty(self):
        self.assertGreaterEqual(len(engine.FREE_GEMINI_MODELS), 1)
        self.assertEqual(engine.FREE_GEMINI_MODELS[0], "gemini-3.6-flash")


class PromptContractTestCase(unittest.TestCase):
    def test_system_prompt_carries_strict_english_directive(self):
        self.assertIn(
            "Analyze the data, translate all context, and generate the final output STRICTLY in English.",
            engine.SYSTEM_PROMPT,
        )
        self.assertIn("macro_score", engine.SYSTEM_PROMPT)
        self.assertIn("black_swan_warnings", engine.SYSTEM_PROMPT)
        self.assertIn("key_drivers", engine.SYSTEM_PROMPT)

    def test_user_prompt_contains_minimal_sections(self):
        ctx = engine.EngineContext(snapshot=_snapshot_vix(21.0))
        system_prompt, user_prompt = engine.build_prompts(ctx)
        self.assertIn("Snapshot: global market data", user_prompt)
        self.assertIn("Recent macro headlines", user_prompt)
        self.assertIn("^VIX", user_prompt)
        self.assertIn("CBOE VIX", user_prompt)


class EndToEndFallbackRunTestCase(unittest.TestCase):
    """No network, no LLM key: the engine must still write a valid JSON file."""

    def test_run_writes_valid_json_without_llm(self):
        ctx_snapshot = _snapshot_vix(28.0)
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sentiment_latest.json"
            with mock.patch.object(engine, "collect_market_snapshot", lambda c: c.snapshot.extend(ctx_snapshot)), \
                 mock.patch.object(engine, "collect_headlines", lambda c: None), \
                 mock.patch.object(engine, "generate_with_llm", side_effect=RuntimeError("no key")):
                code = engine.run(output_path=str(out))
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(list(payload.keys()), CONTRACT_KEYS)
        self.assertIn(payload["market_sentiment"], ("bullish", "bearish", "neutral"))
        self.assertGreaterEqual(payload["macro_score"], -1.0)
        self.assertLessEqual(payload["macro_score"], 1.0)
        joined = json.dumps(payload, ensure_ascii=False)
        self.assertFalse(engine._contains_cjk(joined))

    def test_previous_score_loaded(self):
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sentiment_latest.json"
            out.write_text(json.dumps({"macro_score": 0.61}), encoding="utf-8")
            self.assertEqual(engine.load_previous_score(out), 0.61)

    def test_output_path_resolution_env(self):
        with mock.patch.dict(os.environ, {"MACRO_SENTIMENT_OUTPUT": "custom/feed.json"}, clear=True):
            path = engine.resolve_output_path()
        self.assertEqual(path.name, "feed.json")
        with mock.patch.dict(os.environ, {"MACRO_SENTIMENT_OUTPUT": ""}, clear=True):
            self.assertEqual(engine.resolve_output_path().name, engine.OUTPUT_FILENAME)


class AuditRegressionTestCase(unittest.TestCase):
    """Regression tests for defects caught in the QA audit."""

    def test_partial_market_data_none_change_pct_does_not_crash(self):
        """Yahoo can yield a single close -> change_pct_1d is None (audit bug #1)."""
        snap = [
            {"symbol": "^GSPC", "name": "S&P 500", "role": "equity",
             "last": 6500.0, "change_pct_1d": None},
            {"symbol": "^VIX", "name": "CBOE VIX", "role": "volatility",
             "last": 28.0, "change_pct_1d": None},
        ]
        summary = engine._fallback_summary(snap)
        self.assertIn("US equities", summary)
        ctx = engine.EngineContext(snapshot=snap)
        payload = engine.coerce_payload(engine._rule_based_payload(ctx), ctx)
        engine._validate_contract(payload)

    def test_non_numeric_strings_in_snapshot_do_not_crash(self):
        snap = [
            {"symbol": "^VIX", "name": "CBOE VIX", "role": "volatility",
             "last": "not-a-number", "change_pct_1d": "N/A"},
            {"symbol": "^GSPC", "name": "S&P 500", "role": "equity",
             "last": "6500", "change_pct_1d": "1.2"},
        ]
        ctx = engine.EngineContext(snapshot=snap)
        payload = engine.coerce_payload(engine._rule_based_payload(ctx), ctx)
        engine._validate_contract(payload)

    def test_bool_macro_score_rejected_as_invalid(self):
        payload = engine.coerce_payload(
            {"macro_score": True, "market_sentiment": "neutral"},
            engine.EngineContext(),
        )
        self.assertEqual(payload["macro_score"], 0.0)

    def test_non_string_headline_titles_do_not_crash(self):
        ctx = engine.EngineContext()
        ctx.headlines = [
            {"title": 12345, "source": "", "published": ""},
            {"title": None, "source": "", "published": ""},
            {"title": {"nested": True}, "source": "", "published": ""},
        ]
        payload = engine.coerce_payload(engine._rule_based_payload(ctx), ctx)
        engine._validate_contract(payload)

    def test_run_survives_broken_fallback_and_still_writes_valid_json(self):
        """Sabotage the fallback itself: run() must still emit the contract."""
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sentiment_latest.json"
            with mock.patch.object(engine, "collect_market_snapshot", lambda c: None), \
                 mock.patch.object(engine, "collect_headlines", lambda c: None), \
                 mock.patch.object(engine, "generate_with_llm", side_effect=TimeoutError("llm timeout")), \
                 mock.patch.object(engine, "_rule_based_payload", side_effect=RuntimeError("sabotaged fallback")):
                code = engine.run(output_path=str(out))
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
        engine._validate_contract(payload)
        self.assertEqual(list(payload.keys()), CONTRACT_KEYS)
        self.assertEqual(payload["market_sentiment"], "neutral")
        self.assertEqual(payload["macro_score"], 0.0)

    def test_run_survives_contract_poisoning(self):
        """If coercion itself is sabotaged, minimal contract is written."""
        with TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sentiment_latest.json"
            with mock.patch.object(engine, "collect_market_snapshot", lambda c: None), \
                 mock.patch.object(engine, "collect_headlines", lambda c: None), \
                 mock.patch.object(engine, "generate_with_llm", side_effect=ValueError("bad response")), \
                 mock.patch.object(engine, "coerce_payload", side_effect=RuntimeError("poisoned")):
                code = engine.run(output_path=str(out))
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
        engine._validate_contract(payload)

    def test_validate_contract_rejects_every_deviation(self):
        good = engine._minimal_contract_payload()
        engine._validate_contract(good)

        def expect_reject(mutated, needle=""):
            with self.assertRaises(ValueError) as cm:
                engine._validate_contract(mutated)
            if needle:
                self.assertIn(needle, str(cm.exception))

        extra = dict(good, extra_key=1)
        expect_reject(extra, "keys mismatch")
        bad_sentiment = dict(good, market_sentiment="贪婪")
        expect_reject(bad_sentiment)
        bad_score = dict(good, macro_score=2.0)
        expect_reject(bad_score, "out of [-1.0, 1.0]")
        bad_ts = dict(good, timestamp_utc="yesterday")
        expect_reject(bad_ts)
        bad_list = json.loads(json.dumps(good))
        bad_list["hermes_risk_feed"]["black_swan_warnings"] = "risk"
        expect_reject(bad_list, "list of strings")
        missing_nested = json.loads(json.dumps(good))
        missing_nested["teacher_brain_catalysts"] = {"key_drivers": [], "bonus": 1}
        expect_reject(missing_nested, "keys mismatch")


if __name__ == "__main__":
    unittest.main()
