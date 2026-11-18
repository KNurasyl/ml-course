"""serve --gpu: llama_local takes runtime, gguf and server_args from session_model of route 2.0."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

SOURCE = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


local = load("labs/M09/llama_local.py", "llama_local_gpu")
validator = load("tools/validate_course.py", "validate_course_gpu")


def test_profile_follows_session_model():
    model = yaml.safe_load((SOURCE / "course/route-v2.yaml").read_text(encoding="utf-8"))["session_model"]
    profile = local.session_profile()
    assert profile["name"] == model["gguf"] == "qwen3.5-4b-q4_k_m"
    assert profile["model"] == local.gguf_path(model["gguf"]) and profile["model"].name == "Qwen3.5-4B-Q4_K_M.gguf"
    assert profile["runtime"] == SOURCE / model["runtime"]
    assert profile["ctx"] == 8192 and profile["extra_args"] == ["-ngl", "99"]


def test_server_args_are_parsed(tmp_path):
    (tmp_path / "course").mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata/reference-models.json").write_text(
        json.dumps({"gguf": [{"name": "m", "path": "models/gguf", "file": "m.gguf"}]}), encoding="utf-8")
    route = {"session_model": {"gguf": "m", "runtime": "runtime/x", "server_args": ["-ngl", 20, "--ctx-size", 2048, "-fa", "on"]}}
    (tmp_path / "course/route-v2.yaml").write_text(yaml.safe_dump(route), encoding="utf-8")
    profile = local.session_profile(tmp_path)
    assert (profile["ctx"], profile["extra_args"]) == (2048, ["-ngl", "20", "-fa", "on"])
    route["session_model"]["server_args"] = ["--parallel", 2]
    (tmp_path / "course/route-v2.yaml").write_text(yaml.safe_dump(route), encoding="utf-8")
    with pytest.raises(ValueError, match="--parallel 1"):
        local.session_profile(tmp_path)


class Started(Exception):
    pass


def test_serve_gpu_starts_the_cuda_build(monkeypatch, tmp_path):
    runtime = tmp_path / "b11803-cuda"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/llama-server.exe").write_bytes(b"")
    profile = dict(local.session_profile(), runtime=runtime)
    monkeypatch.setattr(local, "session_profile", lambda root=local.ROOT: profile)
    monkeypatch.setattr(local, "request", lambda *a, **k: (_ for _ in ()).throw(ConnectionError()))
    seen = {}

    def popen(command, **kwargs):
        seen["command"] = command
        raise Started

    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(Started):
        local.serve("ignored", gpu=True, port=18099)
    command = seen["command"]
    assert command[0] == str(runtime / "bin/llama-server.exe")
    assert command[command.index("-m") + 1] == str(profile["model"])
    assert command[command.index("-c") + 1] == "8192" and command[command.index("-ngl") + 1] == "99"
    assert command[command.index("--alias") + 1] == "qwen3.5-4b-q4_k_m" and command.count("--parallel") == 1


def test_missing_cuda_build_points_to_fetch_cuda(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch-cuda"):
        local.server_exe(tmp_path / "none")


def test_launch_mission_gives_the_gpu_command():
    route = validator.load_route_v2(SOURCE)
    mission = route["levels"][0]["lessons"][0]["missions"][0]
    assert mission["id"] == "M09-L07X-M0"
    assert mission["command"] == r".\.venv\Scripts\python.exe labs\M09\llama_local.py serve --gpu"


def test_command_with_a_missing_script_is_reported(monkeypatch):
    route = validator.load_route_v2(SOURCE)
    command = r".\.venv\Scripts\python.exe labs\M09\none.py serve --gpu"
    route["levels"][0]["lessons"][0]["missions"][0]["command"] = command
    monkeypatch.setattr(validator, "load_route_v2", lambda _root: route)
    errors = []
    validator._validate_route_v2(SOURCE, json.loads((SOURCE / "curriculum.json").read_text(encoding="utf-8")), errors)
    assert [e for e in errors if "command" in e] == [
        f"course/route-v2.yaml: mission M09-L07X-M0 command names no existing script: {command}"]
