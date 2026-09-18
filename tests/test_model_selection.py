# -*- coding: utf-8 -*-
"""Free-tier Gemini auto-select + empty-env hardening tests.

Covers the contract introduced with the main-app model migration:

1. TRAP-1 regression: an EMPTY ``GEMINI_MODEL`` (e.g. ``vars.X || secrets.X || ''``
   in a workflow) must never produce a broken ``gemini/`` model name.
2. Auto-select: nothing configured -> free-tier primary + remaining free-tier
   fallbacks, ordered and deterministic, no network at import time.
3. Overrides: ``GEMINI_MODEL`` / ``GEMINI_MODEL_FALLBACK`` / ``LITELLM_MODEL`` /
   ``LITELLM_FALLBACK_MODELS`` / ``MACRO_SENTIMENT_GEMINI_MODELS`` still win.
4. Drift guard: ``src.config`` defaults, ``src.model_selection`` free-tier list,
   the macro engine list, the config registry UI defaults, and the legacy
   inference sites in ``src/services`` all stay in sync.
"""

from __future__ import annotations

import ast
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.config import Config  # noqa: E402
from src.core.config_registry import get_field_definition  # noqa: E402
from src.macro_sentiment import engine  # noqa: E402
from src.model_selection import (  # noqa: E402
    AUTO_SELECT_MAX_FALLBACKS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_MODEL_FALLBACK,
    FREE_GEMINI_MODELS,
    free_gemini_fallback_models,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ids that are retired or paid-only and therefore must never be a default
# (mirrors the engine's registry; kept local so this file has no import cycle).
RETIRED_OR_PAID_ONLY = {
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-3.1-pro-preview",  # left the free tier on 2026-04-01
    "gemini-3-flash-preview",  # stale preview id
}

_KEY_ENV = {
    "GEMINI_API_KEY": "secret-key-value",
    "STOCK_LIST": "600519",
}


def _load_config(env: dict) -> Config:
    """Load Config from an explicit env snapshot, ignoring any local .env file."""
    with patch.dict(os.environ, env, clear=True):
        with patch("src.config.load_dotenv", return_value=False), \
                patch("src.config.dotenv_values", return_value={}), \
                patch("src.config.setup_env"):
            Config._instance = None
            return Config._load_from_env()


def _bare_models(models) -> list:
    return [str(m).split("/", 1)[-1] for m in models]


class EmptyEnvRegressionTests(unittest.TestCase):
    """TRAP-1: os.getenv(name, default) does NOT protect against ''."""

    def test_blank_gemini_model_never_produces_broken_primary(self):
        config = _load_config({**_KEY_ENV, "GEMINI_MODEL": "", "GEMINI_MODEL_FALLBACK": ""})
        self.assertEqual(config.litellm_model, f"gemini/{DEFAULT_GEMINI_MODEL}")
        self.assertNotEqual(config.litellm_model, "gemini/")
        self.assertNotEqual(config.gemini_model, "")
        self.assertNotEqual(config.gemini_model_fallback, "")
        for model in [config.litellm_model, *config.litellm_fallback_models]:
            self.assertRegex(model, r"^gemini/gemini-")

    def test_whitespace_only_gemini_model_is_treated_as_unset(self):
        config = _load_config({**_KEY_ENV, "GEMINI_MODEL": "   ", "GEMINI_MODEL_FALLBACK": "  "})
        self.assertEqual(config.litellm_model, f"gemini/{DEFAULT_GEMINI_MODEL}")
        self.assertEqual(config.litellm_fallback_models[0], f"gemini/{DEFAULT_GEMINI_MODEL_FALLBACK}")

    def test_agent_try_list_never_contains_empty_model(self):
        from src.config import get_effective_agent_models_to_try

        config = _load_config({**_KEY_ENV, "GEMINI_MODEL": ""})
        models = get_effective_agent_models_to_try(config)
        self.assertTrue(models)
        self.assertNotIn("gemini/", models)
        self.assertNotIn("", models)


class AutoSelectDefaultTests(unittest.TestCase):
    """Nothing configured -> deterministic free-tier chain."""

    def test_unset_models_auto_select_free_tier_chain(self):
        config = _load_config(dict(_KEY_ENV))
        self.assertEqual(config.litellm_model, f"gemini/{FREE_GEMINI_MODELS[0]}")
        self.assertEqual(config.gemini_model, DEFAULT_GEMINI_MODEL)
        self.assertEqual(config.gemini_model_fallback, DEFAULT_GEMINI_MODEL_FALLBACK)
        expected_fallbacks = [
            f"gemini/{m}"
            for m in free_gemini_fallback_models(exclude=FREE_GEMINI_MODELS[0])
        ]
        self.assertEqual(config.litellm_fallback_models, expected_fallbacks)
        self.assertTrue(expected_fallbacks)
        self.assertLessEqual(len(expected_fallbacks), AUTO_SELECT_MAX_FALLBACKS)

    def test_auto_select_is_ordered_and_deterministic(self):
        first = _load_config(dict(_KEY_ENV))
        second = _load_config(dict(_KEY_ENV))
        self.assertEqual(first.litellm_model, second.litellm_model)
        self.assertEqual(first.litellm_fallback_models, second.litellm_fallback_models)
        chain = _bare_models([first.litellm_model, *first.litellm_fallback_models])
        self.assertEqual(chain, [m for m in FREE_GEMINI_MODELS if m in set(chain)])
        self.assertEqual(chain[0], FREE_GEMINI_MODELS[0])

    def test_provider_fallback_chain_is_untouched(self):
        """Non-Gemini key inference keeps its historic shape (no Gemini leakage)."""
        config = _load_config({"ANTHROPIC_API_KEY": "sk-ant-test-key-123", "STOCK_LIST": "600519"})
        self.assertEqual(config.litellm_model, "anthropic/claude-sonnet-4-6")
        self.assertEqual(config.litellm_fallback_models, [])

    def test_model_selection_import_is_side_effect_free(self):
        """No network / env sniffing at import time: fresh interpreter sees the same list."""
        import subprocess
        import sys

        code = (
            "import json;"
            "from src.model_selection import FREE_GEMINI_MODELS, DEFAULT_GEMINI_MODEL;"
            "print(json.dumps([FREE_GEMINI_MODELS, DEFAULT_GEMINI_MODEL]))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO_ROOT)},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-400:])
        imported_list, imported_default = ast.literal_eval(proc.stdout.strip())
        self.assertEqual(imported_list, FREE_GEMINI_MODELS)
        self.assertEqual(imported_default, DEFAULT_GEMINI_MODEL)


class ExplicitOverrideTests(unittest.TestCase):
    """Explicit user config always beats auto-select."""

    def test_pinned_primary_and_fallback_are_respected(self):
        config = _load_config(
            {
                **_KEY_ENV,
                "GEMINI_MODEL": "gemini-2.5-flash",
                "GEMINI_MODEL_FALLBACK": "gemini-2.5-flash-lite",
            }
        )
        self.assertEqual(config.litellm_model, "gemini/gemini-2.5-flash")
        self.assertEqual(config.litellm_fallback_models, ["gemini/gemini-2.5-flash-lite"])

    def test_pinned_primary_keeps_single_inherited_fallback(self):
        config = _load_config({**_KEY_ENV, "GEMINI_MODEL": "gemini-2.5-flash"})
        self.assertEqual(config.litellm_model, "gemini/gemini-2.5-flash")
        self.assertEqual(config.litellm_fallback_models, [f"gemini/{DEFAULT_GEMINI_MODEL_FALLBACK}"])

    def test_explicit_litellm_model_short_circuits_inference(self):
        config = _load_config(
            {**_KEY_ENV, "GEMINI_MODEL": "gemini-2.5-flash", "LITELLM_MODEL": "openai/gpt-5.5"}
        )
        self.assertEqual(config.litellm_model, "openai/gpt-5.5")

    def test_explicit_litellm_fallback_list_is_never_augmented(self):
        config = _load_config(
            {**_KEY_ENV, "LITELLM_FALLBACK_MODELS": "deepseek/deepseek-v4-pro"}
        )
        self.assertEqual(config.litellm_fallback_models, ["deepseek/deepseek-v4-pro"])

    def test_provider_prefixed_fallback_value_is_preserved(self):
        config = _load_config(
            {**_KEY_ENV, "GEMINI_MODEL_FALLBACK": "gemini/gemini-2.5-flash-lite"}
        )
        self.assertIn("gemini/gemini-2.5-flash-lite", config.litellm_fallback_models)

    def test_macro_engine_override_still_wins(self):
        env = {
            "GEMINI_API_KEY": "gk",
            "MACRO_SENTIMENT_MODEL": "",
            "GEMINI_MODEL": "",
            "GEMINI_MODEL_FALLBACK": "",
            "MACRO_SENTIMENT_GEMINI_MODELS": "gemini-3.6-flash, gemini-2.5-flash",
        }
        with patch.dict(os.environ, env, clear=True):
            candidates = engine.resolve_llm_candidates()
        self.assertEqual(
            [c["model"] for c in candidates[:2]],
            ["gemini/gemini-3.6-flash", "gemini/gemini-2.5-flash"],
        )


class DriftGuardTests(unittest.TestCase):
    """One free-tier list, shared by every entry point."""

    def test_engine_and_main_app_share_the_same_list_object(self):
        self.assertIs(engine.FREE_GEMINI_MODELS, FREE_GEMINI_MODELS)

    def test_defaults_derive_from_the_shared_list(self):
        self.assertEqual(DEFAULT_GEMINI_MODEL, FREE_GEMINI_MODELS[0])
        self.assertEqual(DEFAULT_GEMINI_MODEL_FALLBACK, FREE_GEMINI_MODELS[1])
        self.assertEqual(FREE_GEMINI_MODELS[0], "gemini-3.6-flash")  # do not break this

    def test_no_retired_or_paid_only_id_is_a_default(self):
        offenders = (set(FREE_GEMINI_MODELS) | {DEFAULT_GEMINI_MODEL, DEFAULT_GEMINI_MODEL_FALLBACK})
        self.assertEqual(offenders & RETIRED_OR_PAID_ONLY, set())

    def test_config_class_defaults_follow_the_constants(self):
        fields = Config.__dataclass_fields__
        self.assertEqual(fields["gemini_model"].default, DEFAULT_GEMINI_MODEL)
        self.assertEqual(fields["gemini_model_fallback"].default, DEFAULT_GEMINI_MODEL_FALLBACK)

    def test_config_registry_ui_defaults_follow_the_constants(self):
        self.assertEqual(get_field_definition("GEMINI_MODEL")["default_value"], DEFAULT_GEMINI_MODEL)
        self.assertEqual(
            get_field_definition("GEMINI_MODEL_FALLBACK")["default_value"],
            DEFAULT_GEMINI_MODEL_FALLBACK,
        )

    def test_legacy_service_inference_falls_back_to_the_constant(self):
        """The three inference sites must not resurrect a hardcoded model id."""
        sites = {
            "src/services/generation_backend_status_service.py": "_infer_legacy_litellm_model",
            "src/services/system_config_service.py": "_infer_setup_legacy_primary_model",
            "src/services/image_stock_extractor.py": "_resolve_vision_model",
        }
        for rel_path, func_name in sites.items():
            source = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
            self.assertNotIn(
                "gemini-3.1-pro-preview",
                source,
                f"{rel_path}: retired default hardcoded again",
            )
            match = re.search(rf"def {re.escape(func_name)}.*?\n(?:.*\n)*?\n\n", source)
            snippet = match.group(0) if match else source
            self.assertTrue(
                "DEFAULT_GEMINI_MODEL" in snippet or "cfg.gemini_model" in snippet,
                f"{rel_path}: {func_name} no longer derives the Gemini default",
            )

    def test_free_tier_ids_are_referenced_in_docs_and_env_template(self):
        for rel_path in ("docs/full-guide.md", "docs/full-guide_EN.md"):
            text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
            self.assertIn(DEFAULT_GEMINI_MODEL, text, rel_path)

    def test_workflow_never_defaults_to_a_retired_model(self):
        """`|| ''` is only safe while config.py treats '' as unset (TRAP-1)."""
        workflow = (REPO_ROOT / ".github" / "workflows" / "00-daily-analysis.yml").read_text(
            encoding="utf-8"
        )
        offending = [
            line.strip()
            for line in workflow.splitlines()
            if "GEMINI_MODEL" in line and not line.strip().startswith("#")
            and (RETIRED_OR_PAID_ONLY & set(re.findall(r"gemini-[a-z0-9.\-]+", line)))
        ]
        self.assertEqual(offending, [])
        config_src = (REPO_ROOT / "src" / "config.py").read_text(encoding="utf-8")
        self.assertIn("os.getenv('GEMINI_MODEL')", config_src)
        self.assertNotIn("os.getenv('GEMINI_MODEL', '", config_src)


if __name__ == "__main__":
    unittest.main()
