"""Register authored local materials without modifying learner progress."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "metadata/material-registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    curriculum = json.loads((ROOT / "curriculum.json").read_text(encoding="utf-8"))
    for item in registry.values():
        item["version"] = "1.0.0"
    for module in curriculum["modules"]:
        mid = module["id"]
        module["material_status"] = "ready"
        registry[mid]["status"] = "ready"
        for lesson in module["lessons"]:
            lid = lesson["id"]
            lesson["status"] = "materials_ready"
            source = ROOT / f"course/lessons/{lid}.md"
            if not source.is_file():
                raise FileNotFoundError(source)
            registry[lid] = {"id":lid,"title":f"{lid}. {lesson['title']}","path":source.relative_to(ROOT).as_posix(),"version":"1.0.0","status":"ready"}
        sid = f"{mid}-SOLUTIONS"
        solution = ROOT / f"solutions/modules/{mid}.md"
        if not solution.exists():
            raise FileNotFoundError(solution)
        registry[sid] = {"id":sid,"title":f"{mid}. Решения для самопроверки","path":solution.relative_to(ROOT).as_posix(),"version":"1.0.0","status":"ready"}
    registry["REFERENCE-PROJECT"] = {"id":"REFERENCE-PROJECT","title":"Эталон голосового тренажёра","path":"projects/interview_trainer/README.md","version":"1.0.0","status":"ready"}
    path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    (ROOT / "curriculum.json").write_text(json.dumps(curriculum,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(f"Registered {len(registry)} materials")


if __name__ == "__main__":
    main()
