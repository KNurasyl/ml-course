"""Russian embedder lab of labs/M10/ru_embedders.py (T32): query set rules, pooling and a run without new models."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m10_ru_embedders", SOURCE / "labs/M10/ru_embedders.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
diag = lab.load("diagnostic_ru")


def test_query_set_covers_every_topic_twice_and_paraphrases_avoid_the_title_word():
    rows = lab.ru_embed_queries()
    assert len(rows) == 52 and len({r["id"] for r in rows}) == 52
    assert [r["style"] for r in rows].count("paraphrase") == [r["style"] for r in rows].count("term") == 24
    assert all(not r["relevant"] for r in rows if r["style"] == "none") and len(lab.NONE) == 4
    titles = {key: title for key, title, _, _ in diag.TOPICS}
    assert set(lab.PAIRS) == {d["document_id"] for d in diag.documents()} == set(titles)
    for topic, (paraphrase, _) in lab.PAIRS.items():
        stem = titles[topic].split()[0].lower()[:5]
        assert stem not in paraphrase.lower(), (topic, stem)


def test_pooling_modes():
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [9.0, 9.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.equal(lab.pool(hidden, mask, "cls"), torch.tensor([[1.0, 0.0]]))
    assert torch.equal(lab.pool(hidden, mask, "mean"), torch.tensor([[2.0, 1.0]]))  # the padded position is ignored
    assert torch.equal(lab.pool(hidden, mask, "lasttoken"), torch.tensor([[9.0, 9.0]]))  # left padding: last is real


def test_encoder_config_reads_pooling_and_prompts(tmp_path):
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling/config.json").write_text(json.dumps({"pooling_mode_lasttoken": True}), encoding="utf-8")
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": {"query": "Instruct: find\nQuery:", "document": ""}}), encoding="utf-8")
    assert lab.encoder_config(tmp_path) == {"pooling": "lasttoken", "query_prompt": "Instruct: find\nQuery:", "document_prompt": ""}
    (tmp_path / "1_Pooling/config.json").write_text(json.dumps({"pooling_mode_cls_token": True}), encoding="utf-8")
    (tmp_path / "config_sentence_transformers.json").unlink()
    assert lab.encoder_config(tmp_path)["pooling"] == "cls" and lab.encoder_config(tmp_path)["query_prompt"] == ""


def test_recall_at_n():
    assert lab.recall_at(["a", "b", "c"], ["c"], 1) == 0 and lab.recall_at(["a", "b", "c"], ["c"], 3) == 1
    assert lab.recall_at(["a", "b"], ["a", "x"], 2) == 0.5


def test_run_without_new_models_reports_lexical_rows_and_missing_models(tmp_path, monkeypatch):
    registry = json.loads((SOURCE / "metadata/reference-models.json").read_text(encoding="utf-8"))
    names = [e["name"] for e in registry["embedders"]]
    assert names == lab.ENCODERS
    fake_root = tmp_path / "root"
    (fake_root / "metadata").mkdir(parents=True)
    (fake_root / "metadata/reference-models.json").write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setattr(lab, "ROOT", fake_root)  # no model files under the fake root: every embedder is missing
    report = lab.run(encoders=lab.ENCODERS, include_minilm=False, out_dir=tmp_path / "out")
    assert report["status"] == {name: "not_downloaded" for name in lab.ENCODERS}
    assert set(report["results"]) == {"lexical_tfidf", "bm25"}
    for result in report["results"].values():
        for group in ["ru-embed-1/paraphrase", "ru-embed-1/term", "diagnostic-1.1.0/diagnostic"]:
            row = result["table"][group]
            assert row["n"] in (24, 12) and 0 <= row["recall_at_1"] <= row["recall_at_3"] <= row["recall_at_5"] <= 1
    assert (tmp_path / "out/report.json").is_file()
    assert "| lexical_tfidf |" in lab.table_text(report)
