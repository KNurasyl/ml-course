"""Second speech configuration (T45): WER edit counting with alternatives, the phrase set, the shared WAV contract of the
optional engines, engine selection in the app and the frozen synthetic run; no speech model needed."""
from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("m15_speech_compare", ROOT / "labs/M15/speech_compare.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
from projects.interview_trainer import speech_alt


def write_wav(path, rate, samples, channels=1):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return path


@pytest.mark.parametrize("ref, hyp, counts", [
    ("я знаю sql", "я знаю", (1, 0, 1, 0)),                       # one deletion
    ("а", "а б в", (2, 0, 0, 2)),                                  # insertions can push WER above 1
    ("время упало с двухсот", "время упала с 200 до", (3, 2, 0, 1)),
    ("Ёлка и елка", "елка и ёлка", (0, 0, 0, 0)),                  # ё counts as е
])
def test_edits_split_the_wer_into_substitutions_deletions_insertions(ref, hyp, counts):
    e = lab.edits(ref, hyp)
    assert (e["errors"], e["s"], e["d"], e["i"]) == counts and e["errors"] / e["words"] == lab.demo.wer(ref, hyp)


def test_lenient_scoring_takes_the_best_reference_written_before_the_run():
    phrase = next(p for p in lab.phrases() if p["id"] == "n01")
    result = lab.score(phrase, "Время ответа упало с 200 до 90 миллисекунд.")
    assert result["strict"]["errors"] == 2 and result["lenient"]["errors"] == 0


def test_phrase_set_has_three_groups_and_alternatives_only_where_spelling_varies():
    items = lab.phrases()
    groups = [p["group"] for p in items]
    assert len(items) == 40 and len({p["id"] for p in items}) == 40
    assert (groups.count("general"), groups.count("numbers"), groups.count("terms")) == (24, 8, 8)
    assert all(len(p["refs"]) == 1 for p in items if p["group"] == "general")
    assert all(len(p["refs"]) == 2 for p in items if p["group"] == "numbers")


def test_the_optional_engines_keep_the_baseline_wav_contract(tmp_path):
    ok = write_wav(tmp_path / "ok.wav", 16000, np.zeros(16000))
    assert speech_alt.check_wav(ok) == 1.0
    with pytest.raises(ValueError, match="16000"):
        speech_alt.check_wav(write_wav(tmp_path / "piper.wav", 22050, np.zeros(22050)))
    with pytest.raises(ValueError, match="120"):
        speech_alt.check_wav(write_wav(tmp_path / "long.wav", 16000, np.zeros(16000 * 121)))
    with pytest.raises(ValueError, match="4000"):
        speech_alt.synthesize_piper("   ", tmp_path / "x.wav")
    with pytest.raises(FileNotFoundError, match="speech-extra"):
        speech_alt.transcribe_whisper(ok, tmp_path / "no-model")


def test_resampling_keeps_duration_and_pitch():
    rate, seconds = 22050, 0.5
    tone = (8000 * np.sin(2 * np.pi * 440 * np.arange(int(rate * seconds)) / rate)).astype(np.int16)
    out = speech_alt.resample_to_16k(tone, rate)
    assert abs(len(out) - 8000) <= 1 and out.dtype == np.int16
    peak = np.argmax(np.abs(np.fft.rfft(out.astype(float)))) * 16000 / len(out)
    assert abs(peak - 440) < 5


def test_own_recordings_are_converted_to_mono_16k(tmp_path):
    stereo = np.zeros((44100, 2), dtype=np.int16)
    source = write_wav(tmp_path / "mic.wav", 44100, stereo.reshape(-1), channels=2)
    target = lab.to_16k(source, tmp_path / "own.wav")
    with wave.open(str(target), "rb") as audio:
        assert (audio.getnchannels(), audio.getframerate(), audio.getnframes()) == (1, 16000, 16000)


def test_the_app_keeps_vosk_and_sapi_by_default_and_can_switch():
    from projects.interview_trainer import app
    from projects.interview_trainer.speech import synthesize_wav, transcribe_wav
    assert app.ENGINES == {"asr": transcribe_wav, "tts": synthesize_wav}
    try:
        engines = app.select_speech("whisper", "piper")
        assert engines["asr"] is speech_alt.transcribe_whisper and engines["tts"] is speech_alt.synthesize_piper
    finally:
        app.select_speech()
    assert app.ENGINES["asr"] is transcribe_wav


REPORT = json.loads((ROOT / "quality/speech/report.json").read_text(encoding="utf-8"))
SUMMARY = {c["cell"]: c for c in lab.summary()["cells"]}


def test_frozen_run_is_marked_synthetic_and_covers_the_whole_set():
    assert REPORT["synthetic_speech"] is True and REPORT["phrases"] == 40 and REPORT["words"] == 280
    assert set(REPORT["cells"]) == {"sapi->vosk", "piper->vosk", "sapi->whisper", "piper->whisper"}
    assert all(len(c["rows"]) == 40 for c in REPORT["cells"].values())


def test_whisper_beats_vosk_on_both_voices_and_piper_is_harder_to_recognise():
    assert SUMMARY["sapi->whisper"]["lenient"] < SUMMARY["sapi->vosk"]["lenient"]
    assert SUMMARY["piper->whisper"]["lenient"] < SUMMARY["piper->vosk"]["lenient"]
    for asr in ("vosk", "whisper"):  # holds for every Piper sample, not only the stored one
        assert min(REPORT["piper_samples"][asr]["lenient"]) > SUMMARY[f"sapi->{asr}"]["lenient"]
    assert max(REPORT["piper_samples"]["whisper"]["lenient"]) < min(REPORT["piper_samples"]["vosk"]["lenient"])


def test_wer_is_the_sum_of_edits_over_reference_words():
    cell = REPORT["cells"]["sapi->whisper"]["strict"]
    assert (cell["s"], cell["d"], cell["i"], cell["words"]) == (27, 3, 3, 280)
    assert cell["wer"] == round((27 + 3 + 3) / 280, 4)


def test_terms_are_the_hardest_group_and_digits_explain_the_whisper_numbers_gap():
    for cell in SUMMARY.values():
        assert cell["groups"]["terms"]["lenient"] > max(cell["groups"]["general"]["lenient"], cell["groups"]["numbers"]["lenient"])
    numbers = SUMMARY["sapi->whisper"]["groups"]["numbers"]
    assert numbers["strict"] > 3 * numbers["lenient"]  # Whisper writes 200 where the reference says двухсот
    vosk = SUMMARY["sapi->vosk"]["groups"]["numbers"]
    assert vosk["strict"] == vosk["lenient"]           # Vosk writes numbers as words


def test_vosk_is_faster_whisper_more_accurate_and_piper_synthesises_fastest():
    assert SUMMARY["sapi->vosk"]["asr_median_rtf"] < SUMMARY["sapi->whisper"]["asr_median_rtf"] < 1
    tts = lab.summary()["tts"]
    assert tts["piper"]["median_seconds"] < tts["sapi"]["median_seconds"] and tts["piper"]["cold_seconds"] > tts["sapi"]["cold_seconds"]


def test_cer_is_softer_than_wer_and_an_inserted_negation_is_caught():
    for cell in SUMMARY.values():
        assert cell["cer_lenient"] < cell["lenient"]
    assert SUMMARY["sapi->vosk"]["negation_mismatches"] == ["t04"]  # NestJS heard as «не та»
    assert lab.negation_mismatch("Повторный запрос не должен создавать заказ.", "Повторный запрос мне должен создавать заказ.")


def test_a_group_without_phrases_reports_no_wer_instead_of_failing():
    rows = [{"group": "general", "score": lab.score(lab.phrases()[0], lab.phrases()[0]["refs"][0])}]
    assert lab.corpus(rows, "lenient", "terms")["wer"] is None and lab.fmt(None) == "—"
    assert lab.corpus(rows, "lenient", "general")["wer"] == 0.0
