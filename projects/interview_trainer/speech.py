"""Offline Windows speech adapters. No cloud speech services are used."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import subprocess
import wave

ROOT = Path(__file__).resolve().parents[2]
_model = None
_model_path = None


def synthesize_wav(text: str, output: Path, voice: str | None = None) -> Path:
    if not text.strip() or len(text) > 4000:
        raise ValueError("Ожидается от 1 до 4000 символов")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # JSON is passed as base64 data, never interpolated into PowerShell code.
    payload = base64.b64encode(json.dumps({"text": text, "output": str(output),
        "voice": voice or "Microsoft Irina Desktop"}, ensure_ascii=False).encode("utf-8")).decode()
    script = """$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Speech
$p = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__PAYLOAD__')) | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $s.SelectVoice($p.voice)
  $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono)
  $s.SetOutputToWaveFile($p.output,$format)
  $s.Speak($p.text)
} finally { $s.Dispose() }
""".replace("__PAYLOAD__", payload)
    command = base64.b64encode(script.encode("utf-16-le")).decode()
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", command],
                            capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError("Windows SAPI не создал WAV: " + result.stderr.decode("utf-8", errors="replace")[-1000:])
    with wave.open(str(output), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError("SAPI вернул неподдерживаемый формат WAV")
    return output


def transcribe_wav(path: Path, model_dir: Path | None = None) -> str:
    from vosk import KaldiRecognizer, Model, SetLogLevel
    global _model, _model_path
    directory = Path(model_dir or ROOT / "models/vosk-model-small-ru-0.22").resolve()
    if not directory.is_dir():
        raise FileNotFoundError("Сначала запусти: python tools/download_models.py --speech")
    SetLogLevel(-1)
    if _model is None or _model_path != directory:
        _model = Model(str(directory))
        _model_path = directory
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError("Нужен WAV: mono, PCM16, 16000 Hz")
        if audio.getnframes() > 16000 * 120:
            raise ValueError("Запись длиннее 120 секунд")
        recognizer = KaldiRecognizer(_model, 16000)
        pieces = []
        while chunk := audio.readframes(4000):
            if recognizer.AcceptWaveform(chunk):
                pieces.append(json.loads(recognizer.Result()).get("text", ""))
        pieces.append(json.loads(recognizer.FinalResult()).get("text", ""))
    return " ".join(piece for piece in pieces if piece)
