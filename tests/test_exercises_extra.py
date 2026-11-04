"""Extra exercises: references pass, empty starters fail, typical mistakes are caught with clear messages."""
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest
import yaml


SOURCE = Path(__file__).resolve().parents[1]
CHECKERS = sorted((SOURCE / "exercises").glob("M??/check_ex*.py"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parts(checker):
    number = checker.stem.removeprefix("check_")
    starter = next(checker.parent.glob(f"{number}_*.py"))
    return number, starter, SOURCE / "solutions/exercises" / checker.parent.name / starter.name


@pytest.mark.parametrize("checker", CHECKERS, ids=lambda p: f"{p.parent.name}-{p.stem}")
def test_reference_passes_starter_fails_and_hidden_cases_exist(checker):
    number, starter, solution = parts(checker)
    check = load(checker, f"check_{checker.parent.name}_{number}").check
    assert check(load(solution, "reference"), 2026) and check(load(solution, "reference"), 7)
    with pytest.raises(NotImplementedError):
        check(load(starter, "starter"), 2026)
    assert (SOURCE / f"tests/hidden/{checker.parent.name}_{number}.json").is_file()
    assert starter.name in (checker.parent / "README.md").read_text(encoding="utf-8")


def broken(tmp_path, solution, old, new):
    text = solution.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / solution.name
    path.write_text(text.replace(old, new), encoding="utf-8")
    return load(path, f"broken_{abs(hash(new))}")


MISTAKES = [
    ("M02", "ex02", "for matrix in matrices:", "for matrix in reversed(matrices):", "сначала поворот"),
    ("M02", "ex02", "range(inner)) for j", "range(inner - 1)) for j", "matmul"),
    ("M02", "ex03", "for row in rows) / n)", "for row in rows) / (n - 1))", "стандартные отклонения"),
    ("M02", "ex03", "return distances.index(min(distances))",
     "return len(distances) - 1 - distances[::-1].index(min(distances))", "меньший индекс"),
    ("M02", "ex03", "return [[(v - m) / s", "return [[v + 0 * (m + s)", "масштабирования"),
]


@pytest.mark.parametrize("module_id,number,old,new,message", MISTAKES, ids=lambda v: str(v)[:20])
def test_typical_mistakes_fail_with_input_expected_and_actual(tmp_path, module_id, number, old, new, message):
    checker = SOURCE / f"exercises/{module_id}/check_{number}.py"
    _, _, solution = parts(checker)
    with pytest.raises(AssertionError) as failure:
        load(checker, f"check_{module_id}_{number}").check(broken(tmp_path, solution, old, new), 2026)
    text = str(failure.value) + "\n".join(getattr(failure.value, "__notes__", []))
    assert message in text and "ожидалось" in text


def test_rolled_out_modules_have_at_least_three_contracts():
    rollout = yaml.safe_load((SOURCE / "metadata/lesson-template-exceptions.yaml").read_text(encoding="utf-8"))["rollout"]
    for module_id in rollout["exercises"]:
        folder = SOURCE / "exercises" / module_id
        assert (folder / "check.py").is_file() and len(list(folder.glob("check_ex*.py"))) >= 2, module_id
