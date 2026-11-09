"""Security boundary of the interview trainer (T53): data neutralisation before prompts, safe links, a trace journal.

Everything that enters a prompt from outside the code is data: the question (a bank or a retrieved question can carry
an indirect injection), retrieved sources, the candidate's answer and the history. Chat templates mark roles with
special tokens, and both the transformers tokenizer and llama-server turn the text <|im_start|> inside a message into
a real role token (found in T40). neutralize() escapes < and > in data, so no tag or role token can be forged.
The model has no tools: its output is one of the bounded choices built by code, and only fixed rubric text is shown.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
ROLE_MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "</s>", "<s>")


def neutralize(text):
    """Data for a prompt: < and > become &lt; and &gt;, so the data cannot close a tag or forge a role token."""
    return text.replace("<", "&lt;").replace(">", "&gt;") if isinstance(text, str) else text


def safe_url(url):
    """Only absolute http(s) links reach the page; javascript:, data: and relative targets become no link."""
    if not isinstance(url, str) or not url:
        return ""
    parsed = urlparse(url.strip())
    return url.strip() if parsed.scheme in ("http", "https") and parsed.netloc and "\\" not in url else ""


@functools.lru_cache(maxsize=1)
def _guardrails():
    spec = importlib.util.spec_from_file_location("trainer_guardrails", ROOT / "labs/M14/guardrails.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan(text):
    """Names of what fires in a text: injection rules of the T41 input rail and chat-template markers; no spans, no text."""
    if not isinstance(text, str) or not text:
        return []
    found = _guardrails().findings(text)
    names = {f"injection:{rule}" for rule in found.get("injection_rule", [])}
    names |= {f"pii:{kind}" for kind in ("email", "phone", "iin", "card") if found.get(kind)}
    names |= {"secret"} if found.get("secret") else set()
    names |= {"role_marker"} if any(marker in text for marker in ROLE_MARKERS) else set()
    return sorted(names)


def fingerprint(text):
    """A candidate's answer in the journal: sha256 prefix and length, never the text."""
    data = (text or "").encode("utf-8")
    return {"sha256": hashlib.sha256(data).hexdigest()[:16], "chars": len(text or "")}


class TraceLog:
    """Append-only JSONL journal of model turns. Without a path the events stay in memory (the last `keep`)."""

    def __init__(self, path=None, keep=200):
        self.path, self.keep, self.events, self.lock = (Path(path) if path else None), keep, [], threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **event):
        event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
        line = json.dumps(event, ensure_ascii=False)
        with self.lock:
            self.events = (self.events + [event])[-self.keep:]
            if self.path:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
        return event


def turn_event(*, session, mode, question, answer, sources, assessment, seconds):
    """What the journal keeps about one answered turn: inputs as fingerprints and findings, the model output, the guards."""
    validation = assessment.get("validation", {})
    return {"event": "answer", "session": hashlib.sha256(session.encode()).hexdigest()[:12], "mode": mode,
            "question": question, "question_findings": scan(question), "answer": fingerprint(answer), "answer_findings": scan(answer),
            "sources": [s.get("id") for s in sources], "source_findings": {s.get("id"): scan(s.get("text", "")) for s in sources if scan(s.get("text", ""))},
            "model": {k: assessment.get("generation", {}).get(k) for k in ("profile", "backend", "input_tokens", "output_tokens", "truncated")},
            "raw_output": assessment.get("raw_output") if assessment.get("selection_policy") == "bounded_model_choice" else None,
            "status": assessment.get("status"), "validation_errors": validation.get("errors", []),
            "display_policy": assessment.get("display_policy"), "fallback_used": assessment.get("fallback_used"),
            "input_rail": assessment.get("input_rail"), "seconds": round(seconds, 3)}
