"""Real local SAPI TTS -> PCM WAV -> Vosk ASR, plus WER/cancellation tests."""
from __future__ import annotations
import json
import re
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def words(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower().replace("ё", "е"))


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        raise ValueError("WER is undefined for an empty reference")
    previous = list(range(len(hyp) + 1))
    for i, expected in enumerate(ref, 1):
        row = [i]
        for j, actual in enumerate(hyp, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (expected != actual)))
        previous = row
    return previous[-1] / len(ref)


def accept_chunk(active_turn: int, chunk_turn: int, cancelled: set[int]) -> bool:
    return chunk_turn == active_turn and chunk_turn not in cancelled


def run() -> dict:
    from projects.interview_trainer.speech import synthesize_wav, transcribe_wav
    out = ROOT / "artifacts" / "M15"
    out.mkdir(parents=True, exist_ok=True)
    reference = "Расскажите про ваш опыт разработки веб сервисов."
    started = time.perf_counter()
    wav_path = synthesize_wav(reference, out / "question.wav")
    tts_seconds = time.perf_counter() - started
    with wave.open(str(wav_path), "rb") as stream:
        rate, channels, width, frames = stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes()
    assert rate == 16000 and channels == 1 and width == 2, "ASR input must be 16 kHz, mono, PCM16"
    started = time.perf_counter()
    hypothesis = transcribe_wav(wav_path)
    asr_seconds = time.perf_counter() - started
    assert wer("я знаю sql", "я знаю") == 1 / 3
    assert wer("а", "а б в") == 2.0
    assert not accept_chunk(8, 7, {7})
    assert accept_chunk(8, 8, {7})
    result = {"reference": reference, "hypothesis": hypothesis,
              "wer": wer(reference, hypothesis), "sample_rate": rate,
              "audio_seconds": frames / rate, "tts_seconds": tts_seconds,
              "asr_seconds": asr_seconds, "asr_rtf": asr_seconds / (frames / rate),
              "cancellation_checks": "passed", "quality_scope": "One synthetic voice sample; not a microphone/noise benchmark"}
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
