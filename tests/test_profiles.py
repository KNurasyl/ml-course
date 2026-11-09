"""Model profiles of the interview trainer (T50): profiles.yaml, readiness checks, no silent fallback, and the LocalLLM
contract over a llama.cpp or generic chat endpoint, checked against a fake server. The live strong profile runs only
with COURSE_STRONG_TESTS=1 and a downloaded GGUF: it starts a 4B model."""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from projects.interview_trainer import profiles
from projects.interview_trainer.llm import LocalLLM
from projects.interview_trainer.llm_server import ServerLLM, choice_grammar, choice_schema

CHOICES = [json.dumps({"evidence_id": i, "criterion": c, "next_step": n}, separators=(",", ":"))
           for i in range(2) for c in ("action", "result") for n in ("reason", "result")]


def test_profiles_file_has_tiny_by_default_and_names_why_others_are_off():
    data = profiles.load_profiles()
    assert data["default"] == "tiny" and set(data["profiles"]) == {"tiny", "strong", "ru-8b", "api"}
    assert all(p.get("context_tokens") == 1800 for p in data["profiles"].values())  # one contract for every profile
    for name in ("ru-8b", "api"):
        assert data["profiles"][name]["enabled"] is False and profiles.check(name, data) == [data["profiles"][name]["reason"]]
    assert data["profiles"]["strong"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_a_disabled_profile_is_refused_and_never_replaced_by_another():
    with pytest.raises(profiles.ProfileUnavailable, match="ru-8b"), profiles.open_model("ru-8b"):
        pass
    with pytest.raises(ValueError, match="unknown profile"):
        profiles.get_profile("huge")


def test_check_lists_what_is_missing(tmp_path, monkeypatch):
    data = {"default": "a", "profiles": {
        "a": {"backend": "transformers", "model_dir": str(tmp_path / "none"), "download": "python tools/download_models.py --llm"},
        "b": {"backend": "openai_compatible", "base_url": None, "model": None, "api_key_env": "T50_TEST_KEY"}}}
    assert profiles.check("a", data) == ["model not downloaded: python tools/download_models.py --llm"]
    monkeypatch.delenv("T50_TEST_KEY", raising=False)
    assert len(profiles.check("b", data)) == 2


def test_tiny_opens_lazily_with_the_profile_values():
    with profiles.open_model("tiny") as model:
        assert isinstance(model, LocalLLM) and model.model is None  # nothing loaded until the first answer
        assert (model.threads, model.context_tokens, model.profile) == (4, 1800, "tiny")
    assert LocalLLM().directory == ROOT / "models/Qwen2.5-0.5B-Instruct"  # the old call keeps working


def test_choice_grammar_allows_exactly_the_literal_strings():
    grammar = choice_grammar(['{"a":"x"}', 'back\\slash'])
    assert grammar == 'root ::= "{\\"a\\":\\"x\\"}" | "back\\\\slash"'


def test_choice_schema_needs_a_product_of_fields():
    schema = choice_schema(CHOICES)
    assert schema["properties"]["evidence_id"]["enum"] == [0, 1] and schema["required"] == ["evidence_id", "criterion", "next_step"]
    with pytest.raises(ValueError, match="product"):
        choice_schema(CHOICES[:-1])


class FakeServer:
    """Records requests; answers /apply-template, /tokenize and /v1/chat/completions like llama-server."""

    def __init__(self, tokens=30, content="{}", finish="stop"):
        self.requests, self.tokens, self.content, self.finish = [], tokens, content, finish
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append((self.path, body, self.headers.get("Authorization")))
                answer = {"/apply-template": {"prompt": "rendered"}, "/tokenize": {"tokens": list(range(fake.tokens))},
                          "/v1/chat/completions": {"choices": [{"message": {"content": fake.content}, "finish_reason": fake.finish}],
                                                   "usage": {"prompt_tokens": fake.tokens, "completion_tokens": 9}}}[self.path]
                data = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake():
    servers = []

    def make(**kwargs):
        servers.append(FakeServer(**kwargs))
        return servers[-1]
    yield make
    for server in servers:
        server.close()


def test_llama_server_choice_counts_first_and_constrains_by_grammar(fake):
    server = fake(content=CHOICES[3])
    model = ServerLLM(server.url, "qwen3.5-4b-q4_k_m", extra={"chat_template_kwargs": {"enable_thinking": False}}, profile="strong")
    assert model.generate_choice([{"role": "user", "content": "x"}], CHOICES) == CHOICES[3]
    paths = [r[0] for r in server.requests]
    assert paths == ["/apply-template", "/tokenize", "/v1/chat/completions"]
    apply, completion = server.requests[0][1], server.requests[2][1]
    assert apply["chat_template_kwargs"] == {"enable_thinking": False} == completion["chat_template_kwargs"]
    assert completion["grammar"] == choice_grammar(CHOICES) and completion["temperature"] == 0 and completion["seed"] == 17
    assert set(contract_keys()) <= set(model.last_generation) and model.last_generation["choice_count"] == len(CHOICES)


def contract_keys():
    return ("input_tokens", "output_tokens", "stop_policy", "choice_count", "truncated", "profile", "backend")


def test_the_context_limit_stops_the_request_before_generation(fake):
    server = fake(tokens=1801)
    with pytest.raises(ValueError, match="1800"):
        ServerLLM(server.url, "m").generate([{"role": "user", "content": "x"}])
    assert "/v1/chat/completions" not in [r[0] for r in server.requests]


def test_a_cut_answer_is_marked_truncated(fake):
    server = fake(content='{"a": ', finish="length")
    model = ServerLLM(server.url, "m")
    assert model.generate_json([{"role": "user", "content": "x"}]) == '{"a":'
    assert model.last_generation["truncated"] is True and model.last_generation["stop_policy"] == "complete_json_or_limit"
    assert server.requests[-1][1]["response_format"] == {"type": "json_object"}


def test_a_generic_endpoint_gets_a_schema_and_the_key_stays_out_of_logs(fake):
    server = fake(content=CHOICES[0], tokens=1801)
    model = ServerLLM(server.url, "paid-model", llama_cpp=False, api_key="secret-test-value", profile="api")
    assert "secret" not in repr(model)
    with pytest.raises(ValueError, match="1800"):  # counted from usage after the call: a generic endpoint has no /tokenize
        model.generate_choice([{"role": "user", "content": "x"}], CHOICES)
    _path, body, auth = server.requests[-1]
    assert [r[0] for r in server.requests] == ["/v1/chat/completions"] and auth == "Bearer secret-test-value"
    assert body["response_format"]["json_schema"]["schema"] == choice_schema(CHOICES) and "grammar" not in body and "seed" not in body


@pytest.mark.skipif(os.environ.get("COURSE_STRONG_TESTS") != "1" or profiles.check("strong") != [],
                    reason="set COURSE_STRONG_TESTS=1 and download the GGUF: starts a 4B model")
def test_strong_profile_runs_the_rubric_end_to_end():
    result = profiles.smoke("strong")
    assert [t["status"] for t in result["turns"]] == ["grounded_candidate", "grounded_candidate"]
    assert all(t["generation"]["backend"] == "llama_server" and not t["generation"]["truncated"] for t in result["turns"])
