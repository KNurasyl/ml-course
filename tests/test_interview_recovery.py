"""HTTP/state recovery tests with a declared model double; no Qwen or speech load."""
from __future__ import annotations

import http.client
import json
from pathlib import Path
import socket
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeModel:
    """Only tests the HTTP contract and state transitions, never model quality."""

    def __init__(self):
        self.model = None
        self.last_generation = {"fixture": True}
        self.calls = 0
        self.fail_next = False

    def generate(self, messages):
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("fixture generation failure")
        return f"Fixture feedback {self.calls}"


@pytest.fixture
def recovery_server(monkeypatch):
    from projects.interview_trainer import app

    fake = FakeModel()
    read_started = threading.Event()
    monkeypatch.setattr(app, "MODEL", fake)
    monkeypatch.setattr(app, "SESSIONS", {})
    monkeypatch.setattr(app, "WORK", threading.Lock())

    class ObservedHandler(app.Handler):
        def setup(self):
            super().setup()
            original = self.rfile

            class ObservedReader:
                def read(self, size=-1):
                    # Signal exactly when body reading begins; avoids sleep races.
                    read_started.set()
                    return original.read(size)

                def __getattr__(self, name):
                    return getattr(original, name)

            self.rfile = ObservedReader()

    server = app.ThreadingHTTPServer(("127.0.0.1", 0), ObservedHandler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    context = SimpleNamespace(
        app=app, fake=fake, port=server.server_port, read_started=read_started
    )
    try:
        yield context
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def request(server, method, path, body=None, *, raw=False):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=3)
    payload = body if raw or body is None else json.dumps(body)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    try:
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def start(server):
    status, state = request(server, "POST", "/start", {"mode": "hr"})
    assert status == 200
    return state


def assert_lock_reusable(server):
    # A bounded acquire allows the request handler to finish its finally block.
    assert server.app.WORK.acquire(timeout=1), "Request left the work lock held"
    server.app.WORK.release()
    assert request(server, "POST", "/start", {"mode": "hr"})[0] == 200


def test_recover_committed_answer_without_resubmitting_generation(recovery_server):
    server = recovery_server
    initial = start(server)
    body = {"id": initial["id"], "turn": 0, "answer": "I implemented an API."}

    # The server commits; simulate a client which did not apply that response.
    assert request(server, "POST", "/answer", body)[0] == 200
    assert initial["turn"] == 0
    assert request(server, "POST", "/answer", body)[0] == 400
    assert server.fake.calls == 1  # Stale retry must not regenerate or duplicate.

    status, recovered = request(server, "GET", f"/session/{initial['id']}")
    assert status == 200
    assert recovered == server.app.SESSIONS[initial["id"]].state()
    assert recovered["turn"] == 1
    assert len(recovered["history"]) == 1
    assert recovered["history"][0]["answer"] == body["answer"]
    assert recovered["history"][0]["feedback"] == "Fixture feedback 1"

    status, continued = request(server, "POST", "/answer", {
        "id": recovered["id"], "turn": recovered["turn"],
        "answer": "I compared two implementation alternatives.",
    })
    assert status == 200
    assert continued["turn"] == 2 and len(continued["history"]) == 2
    assert server.fake.calls == 2
    # The previously delivered JSON snapshot cannot mutate with server history.
    assert recovered["turn"] == 1 and len(recovered["history"]) == 1
    assert_lock_reusable(server)


def test_unknown_session_returns_404_and_releases_lock(recovery_server):
    status, body = request(recovery_server, "GET", "/session/does-not-exist")
    assert status == 404 and "error" in body
    assert recovery_server.fake.calls == 0
    assert_lock_reusable(recovery_server)


def test_stalled_body_does_not_block_start_and_times_out(recovery_server):
    server = recovery_server
    with socket.create_connection(("127.0.0.1", server.port), timeout=7) as stalled:
        stalled.sendall(
            b"POST /start HTTP/1.0\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n{"
        )
        assert server.read_started.wait(timeout=2), "Body read did not begin"
        assert not server.app.WORK.locked(), "Incomplete upload took model lock"
        assert request(server, "POST", "/start", {"mode": "hr"})[0] == 200

        # Exercise the real configured five-second socket timeout, not a mock.
        response = http.client.HTTPResponse(stalled)
        response.begin()
        assert response.status == 408
        assert "error" in json.loads(response.read())
        response.close()
    assert server.fake.calls == 0
    assert_lock_reusable(server)


@pytest.mark.parametrize("bad_turn,current_turn", [
    (False, 0), (True, 1), (0.0, 0), (1.0, 1), ("0", 0), (None, 0),
])
def test_turn_requires_real_int_and_rejection_keeps_state(
    recovery_server, bad_turn, current_turn
):
    server = recovery_server
    state = start(server)
    if current_turn:
        status, state = request(server, "POST", "/answer", {
            "id": state["id"], "turn": 0, "answer": "A first valid answer.",
        })
        assert status == 200
    calls_before = server.fake.calls
    status, error = request(server, "POST", "/answer", {
        "id": state["id"], "turn": bad_turn, "answer": "A valid answer body.",
    })
    assert status == 400 and "error" in error
    assert server.fake.calls == calls_before
    status, recovered = request(server, "GET", f"/session/{state['id']}")
    assert status == 200
    assert recovered["turn"] == current_turn
    assert len(recovered["history"]) == current_turn
    assert_lock_reusable(server)


@pytest.mark.parametrize("body,raw", [(b"{", True), ({"mode": "invalid"}, False)])
def test_parse_and_validation_failures_leave_work_lock_available(
    recovery_server, body, raw
):
    status, error = request(recovery_server, "POST", "/start", body, raw=raw)
    assert status == 400 and "error" in error
    assert recovery_server.app.SESSIONS == {}
    assert_lock_reusable(recovery_server)


def test_failed_generation_can_be_retried_without_advancing_turn(recovery_server):
    server = recovery_server
    state = start(server)
    server.fake.fail_next = True
    body = {"id": state["id"], "turn": 0, "answer": "A valid answer body."}
    assert request(server, "POST", "/answer", body)[0] == 500
    status, recovered = request(server, "GET", f"/session/{state['id']}")
    assert status == 200 and recovered["turn"] == 0 and recovered["history"] == []
    assert_lock_reusable(server)
    status, result = request(server, "POST", "/answer", body)
    assert status == 200 and result["turn"] == 1 and len(result["history"]) == 1


@pytest.mark.parametrize("method,path,body", [
    ("POST", "/start", {"mode": "hr"}),
    ("GET", "/session/anything", None),
])
def test_busy_rejection_does_not_release_another_owners_lock(
    recovery_server, method, path, body
):
    server = recovery_server
    assert server.app.WORK.acquire(blocking=False)
    try:
        status, error = request(server, method, path, body)
        assert status == 409 and "error" in error
        assert server.app.WORK.locked(), "Rejected request released someone else's lock"
    finally:
        server.app.WORK.release()
    assert_lock_reusable(server)
