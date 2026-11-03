"""Integrate reviewed authoring metadata without editing learner attempts."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.1.0"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write(path, data):
    (ROOT / path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    curriculum = read("curriculum.json")
    modules = {m["id"]: m for m in curriculum["modules"]}
    upgrades = [read(f"metadata/upgrade-{name}.json") for name in ("foundations", "llm", "product", "root")]
    product_practice = {
        "M13-L06": "Измерить очередь и дедлайн через настоящий HTTP, воспроизвести перегрузку и восстановление после рестарта.",
        "M14-L06": "Выполнить реальные предложения LLM через валидатор инструментов и сравнить успех задачи с соблюдением доступа.",
        "M15-L06": "Передать PCM фрагментами в ASR, измерить partial/final и отменить очередь TTS с поздним фрагментом.",
        "M16-L06": "Получить обратную связь с цитатами, проверить основания и сравнить исправление на новых ответах.",
        "M16-L07": "Реализовать ограниченные уточнения, повторную доставку и корректное завершение обоих режимов интервью.",
        "M16-L08": "Защитить работающее приложение и новое требование с сохранением состояния после перезапуска.",
    }
    for upgrade in upgrades:
        for lesson in upgrade.get("new_lessons", upgrade.get("lessons", [])):
            lesson = dict(lesson)
            lesson.setdefault("practice", product_practice.get(lesson["id"], ""))
            lesson.setdefault("assessment", f"Выполнить A, новый B и отложенный C по условиям {lesson['id']}; код и объяснение обязательны.")
            lesson.setdefault("path", f"course/lessons/{lesson['id']}.md")
            mid = lesson["id"].split("-")[0]
            module = modules[mid]
            outcome_id = f"{mid}-C{int(lesson['id'].split('L')[1])+10:02}"
            if outcome_id not in {o["id"] for o in module["outcomes"]}:
                module["outcomes"].append({"id": outcome_id, "description": lesson["practice"]})
            item = dict(lesson, outcome_ids=[outcome_id])
            module["lessons"] = [l for l in module["lessons"] if l["id"] != lesson["id"]] + [item]
        for mid, order in upgrade.get("module_order", upgrade.get("module_lesson_orders", {})).items():
            by_id = {l["id"]: l for l in modules[mid]["lessons"]}
            assert set(order) == set(by_id), (mid, set(by_id) ^ set(order))
            modules[mid]["lessons"] = [by_id[lid] for lid in order]
        for lid, deps in upgrade.get("prerequisite_updates", upgrade.get("prerequisite_overrides", {})).items():
            lesson = next(l for l in modules[lid[:3]]["lessons"] if l["id"] == lid)
            lesson["prerequisites"] = deps
    # Hypotheses, not measured learner time. Separate reading, practice and review.
    for mi, module in enumerate(curriculum["modules"]):
        for li, lesson in enumerate(module["lessons"]):
            lid = lesson["id"]
            if int(lid.split("L")[1]) <= 5:
                # Longer integrated numerical tasks take more practice than definitions.
                reading = 25 if mi <= 3 else 35
                practice = 35 + 10 * (mi // 4) + (20 if li in (2, 4) else 0)
                review = 20 if mi <= 5 else 30
                lesson["estimated_minutes"] = reading + practice + review
            else:
                total = lesson.get("estimated_minutes", 180)
                reading, review = min(45, total // 3), 30
                practice = total - reading - review
            lesson["workload"] = {"reading_minutes": reading, "practice_minutes": practice,
                "self_check_minutes": review, "setup_minutes": "separate; record actual",
                "estimate_status": "planning_hypothesis_not_measured"}
            lesson["estimate_status"] = "planning_hypothesis"
            lesson["assessment_policy"] = "course/assessment.md"
        hw = module["homework"]
        hw["canonical_task_and_rubric"] = f"course/modules/{module['id']}.md"
        hw["canonical_solutions"] = f"solutions/modules/{module['id']}.md"
        hw["mandatory_gates"] = ["correctness", "reasoning", "independent_transfer", "no_critical_error"]
        # Avoid a second independently editable version of task conditions.
        hw["task"] = f"Выполни {hw['id']} по актуальному условию в {hw['canonical_task_and_rubric']}."
        hw["retry"] = "Вариант B: отдельное условие и ключ в канонических материалах модуля."
        hw["delayed_check"] = "Вариант C через 3–7 дней: отдельное условие и ключ в материалах модуля."
        hw["estimated_minutes_range"] = [120 + mi * 10, 240 + mi * 20] if mi < 16 else [1800, 3600]
        hw["estimate_status"] = "planning_hypothesis_not_measured"
    m05l02 = next(l for l in modules["M05"]["lessons"] if l["id"] == "M05-L02")
    m05l02["practice"] = "Вычислить вероятности, BCE и матрицу ошибок для готовых scores; обучение классификатора выполнить в M05-L06."
    curriculum["version"] = VERSION
    curriculum["development_note"] = "Самостоятельные материалы дополнены по аудиту A01–A28. Технические проверки и авторские разборы не являются результатами ученика или реальным учебным пилотом."
    curriculum["assessment_policy"] = "course/assessment.md"
    curriculum["checkpoints"] = "course/checkpoints.md"
    write("curriculum.json", curriculum)
    registry = read("metadata/material-registry.json")
    for key, title, path in [
        ("ASSESSMENT", "Как засчитывать самостоятельную работу", "course/assessment.md"),
        ("CHECKPOINTS", "Контрольные точки и возврат к предпосылкам", "course/checkpoints.md"),
        ("CHECKPOINT-SOLUTIONS", "Ключи контрольных точек", "solutions/checkpoints.md"),
        ("EVALUATION-REFERENCE", "Реальный цикл улучшения обратной связи", "course/feedback-case-study.md"),
    ]:
        registry[key] = dict(id=key, title=title, path=path, version=VERSION, status="ready")
    for entry in registry.values():
        entry["version"] = VERSION
    write("metadata/material-registry.json", registry)
    lessons = [l for m in curriculum["modules"] for l in m["lessons"]]
    seen = set()
    for lesson in lessons:
        assert set(lesson["prerequisites"]) <= seen, (lesson["id"], set(lesson["prerequisites"]) - seen)
        seen.add(lesson["id"])
    print(json.dumps({"modules": len(modules), "lessons": len(lessons), "dependency_order": "passed"}))


if __name__ == "__main__":
    main()
