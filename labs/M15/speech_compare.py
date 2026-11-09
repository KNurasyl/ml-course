"""Second speech configuration against the baseline (extension 1.2.0, T45, lesson M15-L07X).

Baseline of M15: Windows SAPI (voice Irina) speaks, Vosk small-ru recognises. Second configuration: Piper (voice
ru_RU-denis-medium, CC0 data) speaks, faster-whisper small in int8 recognises. 40 course-authored interview phrases in
three groups (general, numbers, terms) go through both voices and both recognisers: 2 x 2 cells of WER and latency.

The authored run uses synthetic speech as a stand-in for a person: clean audio, no microphone, no noise, no accent.
It measures the pipeline and the relative behaviour of the engines, not quality on real candidates. The learner's own
recordings go through `own` (task M15-L07X-A) and are reported separately.

WER is counted over the whole set (edits over reference words), strict against the first reference and lenient against
the best of the alternatives written before the run (digits for numbers, Cyrillic spellings for Latin terms).

    python labs/M15/speech_compare.py run                 # needs faster-whisper, piper-tts and --speech-extra models
    python labs/M15/speech_compare.py own recordings\\     # your WAVs and manifest.csv with columns file,id
    python labs/M15/speech_compare.py report              # frozen run in quality/speech, no models
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import statistics
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FROZEN = ROOT / "quality/speech"
OUT = ROOT / "artifacts/M15/speech-compare"
PHRASES = ROOT / "data/speech/phrases-ru.json"
GROUPS = ("general", "numbers", "terms")
# Piper samples noise inside the voice model, so the same text gives different audio on every call. Two extra
# syntheses of the whole set show how much of a Piper result is chance.
PIPER_SAMPLES = 3


def load(rel):
    spec = importlib.util.spec_from_file_location(f"m15_{Path(rel).stem}", ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demo = load("labs/M15/demo.py")


def phrases():
    return json.loads(PHRASES.read_text(encoding="utf-8"))["phrases"]


def edits(reference, hypothesis):
    """Substitutions, deletions and insertions of the minimal alignment; their sum over len(reference) is the WER."""
    ref, hyp = demo.words(reference), demo.words(hypothesis)
    table = [[(j, 0, 0, j) for j in range(len(hyp) + 1)]]  # (cost, S, D, I)
    for i, expected in enumerate(ref, 1):
        row = [(i, 0, i, 0)]
        for j, actual in enumerate(hyp, 1):
            up, left, diag = table[i - 1][j], row[j - 1], table[i - 1][j - 1]
            options = [(up[0] + 1, up[1], up[2] + 1, up[3]), (left[0] + 1, left[1], left[2], left[3] + 1),
                       (diag[0] + (expected != actual), diag[1] + (expected != actual), diag[2], diag[3])]
            row.append(min(options))
        table.append(row)
    cost, s, d, i = table[-1][-1]
    return {"words": len(ref), "s": s, "d": d, "i": i, "errors": cost}


def char_errors(reference, hypothesis):
    """Character edit distance over the normalised words joined by single spaces; CER = errors / reference characters."""
    ref, hyp = " ".join(demo.words(reference)), " ".join(demo.words(hypothesis))
    previous = list(range(len(hyp) + 1))
    for i, expected in enumerate(ref, 1):
        row = [i]
        for j, actual in enumerate(hyp, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (expected != actual)))
        previous = row
    return previous[-1], len(ref)


NEGATIONS = {"не", "нет", "ни"}


def negation_mismatch(reference, hypothesis):
    """A lost or an inserted negation flips the meaning while costing one word of WER."""
    def count(text):
        return sum(w in NEGATIONS for w in demo.words(text))
    return count(reference) != count(hypothesis)


def score(phrase, hypothesis):
    strict = edits(phrase["refs"][0], hypothesis)
    lenient = min((edits(r, hypothesis) for r in phrase["refs"]), key=lambda e: (e["errors"] / e["words"], e["errors"]))
    return {"strict": strict, "lenient": lenient}


def corpus(rows, key, group=None):
    picked = [r["score"][key] for r in rows if group in (None, r["group"])]
    words, errors = sum(p["words"] for p in picked), sum(p["errors"] for p in picked)
    return {"wer": round(errors / words, 4) if words else None, "words": words, "errors": errors,  # None: no phrase of the group
            "s": sum(p["s"] for p in picked), "d": sum(p["d"] for p in picked), "i": sum(p["i"] for p in picked)}


def timing(seconds, audio):
    rtf = [s / a for s, a in zip(seconds, audio)]
    ordered = sorted(seconds)
    return {"median_seconds": round(statistics.median(seconds), 3), "p95_seconds": round(ordered[max(0, int(0.95 * len(ordered)) - 1)], 3),
            "median_rtf": round(statistics.median(rtf), 3), "audio_seconds": round(sum(audio), 1)}


def duration(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def to_16k(source, target):
    """Any PCM16 WAV to the baseline contract: mono, 16 kHz; used for the learner's own recordings."""
    import numpy as np

    from projects.interview_trainer.speech_alt import resample_to_16k
    with wave.open(str(source), "rb") as audio:
        if audio.getsampwidth() != 2:
            raise ValueError(f"{source.name}: нужен PCM16 WAV")
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16).reshape(-1, audio.getnchannels())
        rate = audio.getframerate()
    mono = pcm.mean(axis=1).astype(np.int16) if pcm.shape[1] > 1 else pcm[:, 0]
    with wave.open(str(target), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(resample_to_16k(mono, rate).tobytes())
    return target


def recognisers():
    from projects.interview_trainer.speech import transcribe_wav
    from projects.interview_trainer.speech_alt import transcribe_whisper
    return {"vosk": transcribe_wav, "whisper": transcribe_whisper}


def recognise(engines, sources, items):
    """Each recogniser on each source; the first call of an engine loads the model and is timed apart as the cold start."""
    results = {}
    for asr, transcribe in engines.items():
        started = time.perf_counter()
        transcribe(items[0][1][next(iter(sources))])
        cold = time.perf_counter() - started
        for source in sources:
            rows, seconds, audio = [], [], []
            for phrase, paths in items:
                started = time.perf_counter()
                hypothesis = transcribe(paths[source])
                seconds.append(time.perf_counter() - started)
                audio.append(duration(paths[source]))
                rows.append({"id": phrase["id"], "group": phrase["group"], "ref": phrase["refs"][0], "hyp": hypothesis,
                             "score": score(phrase, hypothesis), "seconds": round(seconds[-1], 3)})
            results[f"{source}->{asr}"] = {"tts": source, "asr": asr, "cold_start_seconds": round(cold, 2), "timing": timing(seconds, audio),
                                           "strict": corpus(rows, "strict"), "lenient": corpus(rows, "lenient"),
                                           "groups": {g: {"strict": corpus(rows, "strict", g)["wer"], "lenient": corpus(rows, "lenient", g)["wer"]}
                                                      for g in GROUPS},
                                           "rows": rows}
    return results


def versions():
    import importlib.metadata as md
    return {name: md.version(name) for name in ("vosk", "faster-whisper", "ctranslate2", "piper-tts", "onnxruntime")}


def run(out_dir=OUT):
    from projects.interview_trainer import speech_alt
    from projects.interview_trainer.speech import synthesize_wav
    out = Path(out_dir)
    items = [(p, {}) for p in phrases()]
    tts = {}
    for name, synthesize in (("sapi", synthesize_wav), ("piper", speech_alt.synthesize_piper)):
        folder = out / "wav" / name
        folder.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        synthesize("Проверка связи.", folder / "warmup.wav")  # Piper loads its voice here; SAPI starts a process on every call
        cold = time.perf_counter() - started
        seconds, audio = [], []
        for phrase, paths in items:
            paths[name] = folder / f"{phrase['id']}.wav"
            started = time.perf_counter()
            synthesize(phrase["refs"][0], paths[name])
            seconds.append(time.perf_counter() - started)
            audio.append(duration(paths[name]))
        tts[name] = {"cold_start_seconds": round(cold, 2), "timing": timing(seconds, audio)}
    cells = recognise(recognisers(), ("sapi", "piper"), items)
    variance = {asr: {"strict": [cells[f"piper->{asr}"]["strict"]["wer"]], "lenient": [cells[f"piper->{asr}"]["lenient"]["wer"]]} for asr in ("vosk", "whisper")}
    for sample in range(2, PIPER_SAMPLES + 1):
        folder = out / "wav" / f"piper-{sample}"
        folder.mkdir(parents=True, exist_ok=True)
        again = [(phrase, {"piper": speech_alt.synthesize_piper(phrase["refs"][0], folder / f"{phrase['id']}.wav")}) for phrase, _ in items]
        for cell in recognise(recognisers(), ("piper",), again).values():
            variance[cell["asr"]]["strict"].append(cell["strict"]["wer"])
            variance[cell["asr"]]["lenient"].append(cell["lenient"]["wer"])
    report = {"task": "T45", "synthetic_speech": True, "phrases": len(items), "words": sum(len(demo.words(p["refs"][0])) for p, _ in items),
              "cpu_count": os.cpu_count(), "whisper": {"model": "faster-whisper-small", "compute_type": "int8", "threads": 4, "beam_size": 5},
              "vosk": {"model": "vosk-model-small-ru-0.22", "threads": 1}, "voices": {"sapi": "Microsoft Irina Desktop", "piper": "ru_RU-denis-medium"},
              "versions": versions(), "tts": tts, "cells": cells, "piper_samples": variance,
              "scope": "Synthetic voices as a stand-in for a person: clean audio, no microphone or noise; not a benchmark on candidates.",
              "collected": time.strftime("%Y-%m-%d")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary(report)


def own(folder, out_dir=ROOT / "artifacts/M15/speech-own"):
    """The learner's recordings: manifest.csv with columns file,id (an id from data/speech/phrases-ru.json)."""
    folder, out = Path(folder), Path(out_dir)
    by_id = {p["id"]: p for p in phrases()}
    (out / "wav").mkdir(parents=True, exist_ok=True)
    with (folder / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    items = [(by_id[row["id"]], {"own": to_16k(folder / row["file"], out / "wav" / f"{row['id']}.wav")}) for row in manifest]
    report = {"task": "T45", "synthetic_speech": False, "phrases": len(items), "cells": recognise(recognisers(), ("own",), items),
              "collected": time.strftime("%Y-%m-%d")}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary(report)


def cer(rows, lenient):
    by_id, errors, chars = {p["id"]: p for p in phrases()}, 0, 0
    for row in rows:
        refs = by_id[row["id"]]["refs"] if lenient else [row["ref"]]
        e, n = min((char_errors(r, row["hyp"]) for r in refs), key=lambda x: x[0] / x[1])
        errors, chars = errors + e, chars + n
    return round(errors / chars, 4)


def summary(report=None):
    if report is None:
        report = json.loads((FROZEN / "report.json").read_text(encoding="utf-8"))
    cells = [{"cell": key, "strict": c["strict"]["wer"], "lenient": c["lenient"]["wer"], "groups": c["groups"],
              "cer_strict": cer(c["rows"], False), "cer_lenient": cer(c["rows"], True),
              "negation_mismatches": [r["id"] for r in c["rows"] if negation_mismatch(r["ref"], r["hyp"])],
              "asr_median_rtf": c["timing"]["median_rtf"], "asr_median_seconds": c["timing"]["median_seconds"],
              "asr_cold_seconds": c["cold_start_seconds"]} for key, c in report["cells"].items()]
    tts = {k: {"median_rtf": v["timing"]["median_rtf"], "median_seconds": v["timing"]["median_seconds"], "cold_seconds": v["cold_start_seconds"]}
           for k, v in report.get("tts", {}).items()}
    return {"task": "T45", "synthetic_speech": report["synthetic_speech"], "phrases": report["phrases"], "cells": cells, "tts": tts,
            "piper_samples": report.get("piper_samples")}


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def table_text(result):
    lines = ["| Голос → распознавание | WER строгий | WER мягкий | CER мягкий | Общие | Числа | Термины | RTF распознавания |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in result["cells"]:
        g = c["groups"]
        lines.append(f"| {c['cell']} | {c['strict']:.3f} | {c['lenient']:.3f} | {c['cer_lenient']:.3f} | {fmt(g['general']['lenient'])} | "
                     f"{fmt(g['numbers']['lenient'])} | {fmt(g['terms']['lenient'])} | {c['asr_median_rtf']:.2f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "own", "report"])
    parser.add_argument("folder", nargs="?", type=Path, help="for own: folder with WAV files and manifest.csv")
    args = parser.parse_args()
    if args.command == "own" and not args.folder:
        parser.error("own needs a folder with manifest.csv")
    result = run() if args.command == "run" else own(args.folder) if args.command == "own" else summary()
    print(table_text(result))
    print(json.dumps(result["tts"], ensure_ascii=False))
