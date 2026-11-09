"""Optional second speech configuration (extension 1.2.0, T45): faster-whisper ASR and Piper TTS, local and offline.

Vosk and Windows SAPI in speech.py stay the default. These functions keep the same WAV contract (mono PCM16 16 kHz,
at most 120 seconds, 1 to 4000 characters of text), so the app can swap them with --asr whisper and --tts piper.
Packages: faster-whisper and piper-tts from requirements-optional.in; models: tools/download_models.py --speech-extra.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_whisper = _whisper_path = None
_piper = _piper_path = None


def _models():
    return {m["name"]: m for m in json.loads((ROOT / "metadata/reference-models.json").read_text(encoding="utf-8"))["speech_extra"]}


def whisper_dir() -> Path:
    return ROOT / _models()["faster-whisper-small"]["path"]


def piper_voice() -> Path:
    config = _models()["ru_RU-denis-medium"]
    return ROOT / config["path"] / next(f for f in config["files"] if f.endswith(".onnx"))


def check_wav(path: Path) -> float:
    """The baseline contract: mono, PCM16, 16 kHz, at most 120 seconds. Returns the duration in seconds."""
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError("Нужен WAV: mono, PCM16, 16000 Hz")
        if audio.getnframes() > 16000 * 120:
            raise ValueError("Запись длиннее 120 секунд")
        return audio.getnframes() / 16000


def transcribe_whisper(path: Path, model_dir: Path | None = None, beam_size: int = 5, threads: int = 4) -> str:
    global _whisper, _whisper_path
    directory = Path(model_dir or whisper_dir()).resolve()
    if not (directory / "model.bin").is_file():
        raise FileNotFoundError("Сначала запусти: python tools/download_models.py --speech-extra")
    check_wav(path)
    if _whisper is None or _whisper_path != directory:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(str(directory), device="cpu", compute_type="int8", cpu_threads=threads)
        _whisper_path = directory
    segments, _ = _whisper.transcribe(str(path), language="ru", beam_size=beam_size, condition_on_previous_text=False)
    return " ".join(segment.text.strip() for segment in segments).strip()


def resample_to_16k(pcm, rate: int):
    """Polyphase resampling of int16 samples to 16 kHz; Piper voices speak at 22 050 Hz."""
    import numpy as np
    from scipy.signal import resample_poly
    if rate == 16000:
        return pcm
    divisor = np.gcd(16000, rate)
    audio = resample_poly(pcm.astype(np.float32), 16000 // divisor, rate // divisor)
    return np.clip(np.round(audio), -32768, 32767).astype(np.int16)


def synthesize_piper(text: str, output: Path, voice_path: Path | None = None) -> Path:
    global _piper, _piper_path
    import numpy as np
    if not text.strip() or len(text) > 4000:
        raise ValueError("Ожидается от 1 до 4000 символов")
    voice = Path(voice_path or piper_voice()).resolve()
    if not voice.is_file():
        raise FileNotFoundError("Сначала запусти: python tools/download_models.py --speech-extra")
    if _piper is None or _piper_path != voice:
        from piper import PiperVoice
        _piper, _piper_path = PiperVoice.load(voice), voice
    chunks = list(_piper.synthesize(text))
    if not chunks:
        raise RuntimeError("Piper не вернул звук")
    pcm = np.concatenate([np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16) for chunk in chunks])
    pcm = resample_to_16k(pcm, chunks[0].sample_rate)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(pcm.tobytes())
    return output
