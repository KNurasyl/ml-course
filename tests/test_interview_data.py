import json
from pathlib import Path


def test_interview_tracks_are_separate_and_explicitly_synthetic():
    directory = Path(__file__).resolve().parents[1] / "data/interview"
    all_ids = []
    for name, track, minimum in [("hr-development.json", "hr", 8), ("backend-development.json", "backend", 12)]:
        data = json.loads((directory / name).read_text(encoding="utf-8"))
        assert data["track"] == track
        assert data["status"] == "author_seed_not_human_validated"
        assert len(data["cases"]) >= minimum
        assert data["rubric"]["dimensions"]
        for case in data["cases"]:
            all_ids.append(case["id"])
            assert case["question"]
            assert case.get("evidence_to_seek") or case.get("expected_concepts")
    assert len(set(all_ids)) == len(all_ids)
