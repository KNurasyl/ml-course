"""The LocalLLM contract over an OpenAI-compatible chat endpoint (T50): generate, generate_json, generate_choice.

With a llama.cpp server the limits stay exact: the prompt is counted by /apply-template and /tokenize before generation,
and bounded choices are enforced by a GBNF grammar of the literal choice strings. A generic endpoint (the paid `api`
profile) gets JSON mode and a JSON Schema of per-field enums instead; it counts the prompt only after the call.
Failures stay visible: nothing here substitutes canned feedback for a model answer.
"""
from __future__ import annotations

import itertools
import json
import threading
import time
import urllib.request


def gbnf_literal(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r") + '"'


def choice_grammar(choices) -> str:
    """root ::= "choice 1" | "choice 2" | ...: the model can emit exactly one of the strings and nothing else."""
    return "root ::= " + " | ".join(gbnf_literal(c) for c in choices)


def choice_schema(choices) -> dict:
    """A JSON Schema of per-key enums when the choices are exactly the product of their fields, as in feedback.py."""
    values = [json.loads(c) for c in choices]
    if not all(isinstance(v, dict) for v in values) or len({tuple(v) for v in values}) != 1:
        raise ValueError("choices are not objects with the same keys; a generic endpoint needs a grammar")
    keys = list(values[0])
    fields = {k: list(dict.fromkeys(v[k] for v in values)) for k in keys}
    if {tuple(v[k] for k in keys) for v in values} != set(itertools.product(*fields.values())):
        raise ValueError("choices are not a product of field values; a generic endpoint needs a grammar")
    return {"type": "object", "additionalProperties": False, "required": keys, "properties": {k: {"enum": fields[k]} for k in keys}}


class ServerLLM:
    """One chat endpoint, one request at a time, greedy decoding; last_generation has the keys of LocalLLM."""

    def __init__(self, base_url: str, model: str, *, context_tokens: int = 1800, llama_cpp: bool = True,
                 api_key: str | None = None, extra: dict | None = None, timeout: float = 600, profile: str = ""):
        self.base_url, self.model, self.context_tokens, self.llama_cpp = base_url.rstrip("/"), model, context_tokens, llama_cpp
        self._api_key, self.extra, self.timeout, self.profile = api_key, dict(extra or {}), timeout, profile
        self.lock = threading.Lock()
        self.last_generation = {}

    def __repr__(self):  # the key never appears in logs or tracebacks
        return f"ServerLLM(profile={self.profile!r}, base_url={self.base_url!r}, model={self.model!r})"

    def _post(self, path: str, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        request = urllib.request.Request(self.base_url + path, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def count_tokens(self, messages) -> int | None:
        if not self.llama_cpp:
            return None
        payload = {"messages": messages}
        if "chat_template_kwargs" in self.extra:
            payload["chat_template_kwargs"] = self.extra["chat_template_kwargs"]
        prompt = self._post("/apply-template", payload)["prompt"]
        return len(self._post("/tokenize", {"content": prompt, "add_special": False})["tokens"])

    def generate_json(self, messages, max_new_tokens=192):
        return self.generate(messages, max_new_tokens=max_new_tokens, json_object=True)

    def generate_choice(self, messages, choices):
        if not choices or len(choices) > 100 or not all(isinstance(c, str) and c for c in choices):
            raise ValueError("1..100 explicit string choices required")
        return self.generate(messages, max_new_tokens=256, choices=choices)

    def generate(self, messages, max_new_tokens=128, json_object=False, choices=None):
        with self.lock:
            counted = self.count_tokens(messages)
            if counted is not None and counted > self.context_tokens:
                raise ValueError(f"Контекст превышает учебный лимит {self.context_tokens} токенов: сократи ответ")
            payload = {"model": self.model, "messages": messages, "max_tokens": max_new_tokens, "temperature": 0, **self.extra}
            if self.llama_cpp:
                payload.update(seed=17, cache_prompt=False)
            if choices is not None:
                if self.llama_cpp:
                    payload["grammar"] = choice_grammar(choices)
                else:
                    payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "choice", "strict": True, "schema": choice_schema(choices)}}
            elif json_object:
                payload["response_format"] = {"type": "json_object"}
            started = time.perf_counter()
            body = self._post("/v1/chat/completions", payload)
            seconds = time.perf_counter() - started
            choice = body["choices"][0]
            text = (choice["message"].get("content") or "").strip()
            usage = body.get("usage") or {}
            if counted is None and usage.get("prompt_tokens", 0) > self.context_tokens:
                raise ValueError(f"Контекст превышает учебный лимит {self.context_tokens} токенов: сократи ответ")
            self.last_generation = {
                "input_tokens": int(usage.get("prompt_tokens", counted or 0)), "output_tokens": int(usage.get("completion_tokens", 0)),
                "stop_policy": "bounded_token_choices" if choices is not None else ("complete_json_or_limit" if json_object else "eos_or_limit"),
                "choice_count": len(choices) if choices is not None else None, "truncated": choice.get("finish_reason") == "length",
                "profile": self.profile, "backend": "llama_server" if self.llama_cpp else "openai_compatible", "seconds": round(seconds, 3)}
            return text
