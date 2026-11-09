"""Run with: python -m projects.interview_trainer.app (localhost only)."""
from __future__ import annotations
import argparse
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import wave
from .core import Interview, make_prompt
from .llm import LocalLLM
from .feedback import evaluate_answer
from .speech import synthesize_wav, transcribe_wav
from .profiles import open_model

ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).parent / "web"
SESSIONS: dict[str, Interview] = {}
WORK = threading.Lock()
MODEL = LocalLLM()
ENGINES = {"asr": transcribe_wav, "tts": synthesize_wav}


def select_speech(asr: str = "vosk", tts: str = "sapi") -> dict:
    """Vosk and SAPI by default; the optional second configuration of M15-L07X keeps the same WAV contract."""
    from . import speech_alt
    ENGINES["asr"] = {"vosk": transcribe_wav, "whisper": speech_alt.transcribe_whisper}[asr]
    ENGINES["tts"] = {"sapi": synthesize_wav, "piper": speech_alt.synthesize_piper}[tts]
    return ENGINES


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, format, *args):
        # Do not put candidate answers in console logs.
        pass

    def send(self, status, body, content_type="application/json; charset=utf-8"):
        payload = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_GET(self):
        if self.path == "/health":
            return self.send(200, {"status":"ok","profile":getattr(MODEL, "profile", "tiny"),"model_loaded":MODEL.model is not None,"sessions":len(SESSIONS)})
        if self.path.startswith("/session/"):
            if not WORK.acquire(blocking=False):
                return self.send(409, {"error":"Операция ещё выполняется. Повтори восстановление позже."})
            try:
                interview = SESSIONS.get(self.path[len("/session/"):])
                state = json.dumps(interview.state(), ensure_ascii=False).encode("utf-8") if interview else None
            finally:
                WORK.release()
            return self.send(200, state) if state else self.send(404, {"error":"Сессия не найдена. Возможно, сервер перезапущен."})
        if self.path in ("/", "/index.html", "/app.js", "/style.css", "/pcm-worklet.js"):
            name = "index.html" if self.path == "/" else self.path[1:]
            mime = {".html":"text/html; charset=utf-8", ".js":"text/javascript; charset=utf-8", ".css":"text/css; charset=utf-8"}[Path(name).suffix]
            return self.send(200, (WEB / name).read_bytes(), mime)
        self.send(404, {"error":"Не найдено"})

    def body_limit(self):
        return 3_900_000 if self.path == "/asr" else 20000

    def discard_body(self):
        # An early reply must still read the body: closing a socket with unread data makes
        # Windows send RST, and the client may lose the reply. Oversized bodies are not read.
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if 0 < size <= self.body_limit():
                self.rfile.read(size)
        except (ValueError, OSError):
            pass

    def do_POST(self):
        expected = f"http://127.0.0.1:{self.server.server_port}"
        if self.headers.get("Origin") not in (None, expected, f"http://localhost:{self.server.server_port}"):
            self.discard_body()
            return self.send(403, {"error":"Допускается только локальный интерфейс"})
        locked = False
        try:
            size = int(self.headers.get("Content-Length", "0"))
            limit = self.body_limit()
            if not 0 < size <= limit:
                raise ValueError("Пустой или слишком большой запрос")
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError("Тело запроса получено не полностью")
            data = None
            if self.path != "/asr":
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError("Тело запроса должно быть JSON-объектом")
            if not WORK.acquire(blocking=False):
                return self.send(409, {"error":"Модель занята. Дождись результата текущего запроса."})
            locked = True
            if self.path == "/asr":
                with tempfile.TemporaryDirectory(prefix="course-asr-") as directory:
                    audio = Path(directory) / "input.wav"
                    audio.write_bytes(body)
                    start = time.perf_counter()
                    return self.send(200, {"text":ENGINES["asr"](audio),"seconds":time.perf_counter()-start})
            if self.path == "/start":
                if len(SESSIONS) >= 100:
                    raise ValueError("Достигнут учебный лимит сессий: перезапусти сервер")
                interview = Interview(data.get("mode", ""))
                SESSIONS[interview.id] = interview
                return self.send(200, interview.state())
            if self.path == "/answer":
                interview = SESSIONS.get(data.get("id"))
                if interview is None:
                    raise ValueError("Сессия не найдена")
                question, sources = interview.prepare(data.get("answer"), data.get("turn"))
                start = time.perf_counter()
                assessment = evaluate_answer(MODEL, mode=interview.mode, question=question,
                    answer=data['answer'], sources=sources, history=interview.history, config='candidate')
                feedback = assessment['display_text']
                if not assessment['raw_output'].strip():
                    raise RuntimeError("Модель вернула пустой разбор; ход не сохранён")
                result = interview.commit(data["answer"], feedback, sources, time.perf_counter()-start, MODEL.last_generation, assessment)
                # In-memory only; user explicitly exports a session from the browser.
                return self.send(200, result)
            if self.path == "/tts":
                with tempfile.TemporaryDirectory(prefix="course-tts-") as directory:
                    output = ENGINES["tts"](data.get("text", ""), Path(directory) / "reply.wav")
                    return self.send(200, output.read_bytes(), "audio/wav")
            self.send(404, {"error":"Не найдено"})
        except TimeoutError:
            self.send(408, {"error":"Превышено время получения запроса. Повтори отправку."})
        except (ValueError, TypeError, KeyError, wave.Error, EOFError) as error:
            self.send(400, {"error":str(error)})
        except Exception as error:
            self.send(500, {"error":f"{type(error).__name__}: {error}"})
        finally:
            if locked:
                WORK.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--asr", choices=["vosk", "whisper"], default="vosk", help="whisper: optional faster-whisper, M15-L07X")
    parser.add_argument("--tts", choices=["sapi", "piper"], default="sapi", help="piper: optional Piper voice, M15-L07X")
    parser.add_argument("--profile", default=None, help="model profile from profiles.yaml: tiny (default), strong, ru-8b, api")
    args = parser.parse_args()
    select_speech(args.asr, args.tts)
    global MODEL
    with contextlib.ExitStack() as stack:
        MODEL = stack.enter_context(open_model(args.profile))  # a llama.cpp server started here stops with the app
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        server.timeout = 1
        print(f"Профиль модели: {MODEL.profile}. Открой http://127.0.0.1:{args.port} ; Ctrl+C останавливает сервер", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
