"""llama.cpp serving lab (T30): stream parsing, comparison helpers, pinned files and a run without binary or models."""
from __future__ import annotations

import importlib.util
import json
import re
import zipfile
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(f"t30_{name}", SOURCE / f"labs/M09/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab, local = load("llama_serving"), load("llama_local")


def event(payload):
    return "data: " + json.dumps(payload)


def test_parse_sse_takes_first_content_time_text_timings_and_reasoning():
    ticks = iter([10.0, 11.0, 12.0])
    lines = [": comment", event({"choices": [{"delta": {"role": "assistant"}}]}),
             event({"choices": [{"delta": {"reasoning_content": "думаю"}}]}),
             event({"choices": [{"delta": {"content": "При"}}]}), event({"choices": [{"delta": {"content": "вет"}}]}),
             event({"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": {"prompt_n": 5, "predicted_n": 2}}),
             event({"choices": [], "usage": {"completion_tokens": 2}}), "data: [DONE]", event({"choices": [{"delta": {"content": "!"}}]})]
    result = lab.parse_sse([line.encode("utf-8") for line in lines], clock=lambda: next(ticks))
    assert result["text"] == "Привет" and result["first"] == 10.0  # the role-only delta is not the first token
    assert result["timings"] == {"prompt_n": 5, "predicted_n": 2} and result["usage"] == {"completion_tokens": 2}
    assert result["reasoning_chars"] == len("думаю")


def test_common_prefix_and_summary_against_float32():
    assert lab.common_prefix("индекс ускоряет", "индекс замедляет") == 7 and lab.common_prefix("", "a") == 0
    answer = {"ttft_ms": 100.0, "prefill_tps": 50.0, "decode_tps": 20.0, "generated_tokens": 10}
    reference = {"answers": [{**answer, "text": "a b"}, {**answer, "text": "c d"}]}
    result = {"answers": [{**answer, "text": "a b"}, {**answer, "text": "c x", "ttft_ms": 300.0}]}
    summary = lab.summarize(result, reference)
    assert summary["same_text_as_f32"] == 1 and summary["common_prefix_chars_median"] == 2.5
    assert summary["ttft_ms_median"] == 200.0 and summary["generated_tokens_total"] == 20
    assert "same_text_as_f32" not in lab.summarize(result)


def test_gguf_files_and_llama_cpp_release_are_pinned():
    entries = {e["name"]: e for e in json.loads((SOURCE / "metadata/reference-models.json").read_text(encoding="utf-8"))["gguf"]}
    assert {entry for _, runtime, entry in lab.CONFIGS if runtime == "llama.cpp"} == set(entries)
    for entry in entries.values():
        assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        assert entry["file"].endswith(".gguf") and entry["license"] == "Apache-2.0" and entry["bytes"] > 0
    assert "community" in entries["qwen3.5-4b-q4_k_m"]["source"]  # no official GGUF: the provenance must say so
    assert re.fullmatch(r"b\d+", local.RELEASE["tag"]) and re.fullmatch(r"[0-9a-f]{64}", local.RELEASE["sha256"])
    assert local.HOST == "127.0.0.1"


def gguf_bytes():
    """A minimal GGUF v3: two metadata keys, an F32 tensor of 8 weights and a Q8_0 tensor of 32 weights."""
    import struct

    def string(s):
        return struct.pack("<Q", len(s.encode())) + s.encode()

    header = b"GGUF" + struct.pack("<IQQ", 3, 2, 3)
    header += string("general.architecture") + struct.pack("<I", 8) + string("toy")
    header += string("general.alignment") + struct.pack("<II", 4, 32)
    header += string("toy.block_count") + struct.pack("<II", 4, 2)
    header += string("norm") + struct.pack("<I", 1) + struct.pack("<Q", 8) + struct.pack("<IQ", 0, 0)
    header += string("weight") + struct.pack("<I", 2) + struct.pack("<QQ", 4, 8) + struct.pack("<IQ", 8, 32)
    padding = b"\0" * ((32 - len(header) % 32) % 32)
    return header + padding + b"\1" * 32 + b"\2" * 34  # 8 × 4 bytes of F32, one Q8_0 block of 34 bytes


def test_gguf_summary_reads_metadata_and_bits_per_weight(tmp_path):
    path = tmp_path / "toy.gguf"
    path.write_bytes(gguf_bytes())
    summary = lab.gguf_summary(path)
    assert summary["architecture"] == "toy" and summary["layers"] == 2 and summary["tensors"] == 2 and summary["weights"] == 40
    assert summary["types"]["F32"] == {"tensors": 1, "weights": 8, "bytes": 32, "bits_per_weight": 32.0}
    assert summary["types"]["Q8_0"]["bits_per_weight"] == 8.5  # 32 int8 values plus one fp16 scale per block
    assert summary["bits_per_weight"] == round(66 * 8 / 40, 2)
    (tmp_path / "bad.gguf").write_bytes(b"GGML" + gguf_bytes()[4:])
    with pytest.raises(ValueError, match="not a GGUF"):
        lab.gguf_summary(tmp_path / "bad.gguf")


def test_fetch_rejects_an_archive_with_another_checksum(tmp_path, monkeypatch):
    archive = tmp_path / "llama.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("llama-server.exe", b"not a binary")
    monkeypatch.setattr(local, "BIN", tmp_path / "runtime/b0")
    monkeypatch.setattr(local, "RELEASE", {**local.RELEASE, "url": archive.as_uri()})
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        local.fetch()
    assert not (tmp_path / "runtime/b0/llama-server.exe").exists()
    with pytest.raises(FileNotFoundError, match="llama_local.py fetch"):
        local.server_exe()


def test_run_without_models_reports_status_and_writes_report(tmp_path, monkeypatch):
    fake = tmp_path / "root"
    (fake / "metadata").mkdir(parents=True)
    (fake / "metadata/reference-models.json").write_text((SOURCE / "metadata/reference-models.json").read_text(encoding="utf-8"),
                                                         encoding="utf-8")
    monkeypatch.setattr(lab, "ROOT", fake)
    report = lab.run(out_dir=tmp_path / "out")
    assert {r["status"] for r in report["results"].values()} == {"not_downloaded"}
    assert report["llama_cpp"]["tag"] == local.RELEASE["tag"] and len(report["prompts"]) == 6
    assert (tmp_path / "out/report.json").is_file() and "not_downloaded" in lab.table_text(report)


def test_live_server_answers_through_the_openai_api_when_present():
    entry = lab.gguf_entry("qwen2.5-0.5b-instruct-q4_k_m")
    model = SOURCE / entry["path"] / entry["file"]
    try:
        local.server_exe()
    except FileNotFoundError:
        pytest.skip("llama.cpp binary not fetched")
    if not model.exists():
        pytest.skip("GGUF not downloaded")
    with local.Server(model, threads=2, ctx=1024, alias="pytest", log=SOURCE / "artifacts/M09/llama/pytest.server.log"):
        with local.request("/v1/models") as response:
            assert [m["id"] for m in json.load(response)["data"]] == ["pytest"]
        answer = lab.chat_stream(local, "Say OK.", model="pytest")
    assert answer["text"] and answer["ttft_ms"] > 0 and answer["generated_tokens"] >= 1 and answer["decode_tps"] > 0
