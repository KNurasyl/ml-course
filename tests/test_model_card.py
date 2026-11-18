"""M09-L09X lab: card metadata, KV cache per token and the tokenizer tax from local files only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m09_model_card", SOURCE / "labs/M09/model_card.py")
card = importlib.util.module_from_spec(spec)
spec.loader.exec_module(card)


def test_kv_formula_matches_the_lesson_and_the_r06_measurement():
    assert card.kv_bytes_per_token(24, 2, 64) == 12_288  # Qwen2.5-0.5B
    assert card.kv_bytes_per_token(8, 4, 256) == 32 * 1024  # Qwen3.5-4B, 8 full-attention layers
    assert card.kv_bytes_per_token(30, 3, 64) == 23_040  # SmolLM2-135M


def test_front_matter_is_read_from_a_card(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("---\nlicense: apache-2.0\nlanguage:\n- en\nbase_model: x/y\n---\n\n# Model\n", encoding="utf-8")
    assert card.front_matter(readme) == {"license": "apache-2.0", "language": ["en"], "base_model": "x/y"}
    readme.write_text("# No metadata\n", encoding="utf-8")
    assert card.front_matter(readme) == {}


def test_local_qwen_card_and_config_disagree_where_the_lesson_says():
    folder = SOURCE / "models/Qwen2.5-0.5B-Instruct"
    if not (folder / "config.json").is_file():
        import pytest
        pytest.skip("local model not downloaded")
    row = card.hf_model(folder)
    assert row["license"] == "apache-2.0" and row["language"] == ["en"] and row["context"] == 32_768
    assert row["kv_mib_at_context"] == 384.0
    assert row["ru_tokens_per_word"] > 2 * row["en_tokens_per_word"]
