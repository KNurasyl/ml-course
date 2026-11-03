"""Execute pilot notebooks in fresh kernels without changing source notebooks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["starter", "solutions", "all"], default="all")
    parser.add_argument("--lesson", help="Only one lesson, e.g. P03")
    parser.add_argument("--html", action="store_true", help="Also render portable HTML previews")
    args = parser.parse_args()
    folders = []
    if args.kind in ("starter", "all"):
        folders.append(("starter", ROOT / "notebooks" / "pilot"))
    if args.kind in ("solutions", "all"):
        folders.append(("solutions", ROOT / "solutions" / "pilot"))
    reports = []
    for kind, folder in folders:
        sources = sorted(folder.glob(f"{args.lesson or 'P*'}.ipynb"))
        if not sources:
            raise SystemExit(f"No notebooks found: {folder} / {args.lesson or 'P*'}")
        for source in sources:
            destination = ROOT / "artifacts" / "executed" / kind / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            notebook = nbformat.read(source, as_version=4)
            nbformat.validate(notebook)
            manager = KernelManager(kernel_name="python3")
            manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            client = NotebookClient(notebook, km=manager, timeout=120, resources={"metadata": {"path": str(ROOT)}})
            try:
                client.execute()
                nbformat.write(notebook, destination)
                if args.html:
                    from nbconvert import HTMLExporter
                    content, _ = HTMLExporter().from_notebook_node(notebook)
                    destination.with_suffix(".html").write_text(content, encoding="utf-8")
                reports.append({"lesson": source.stem, "kind": kind, "status": "executed", "source": str(source.relative_to(ROOT)), "output": str(destination.relative_to(ROOT))})
                print(f"OK {kind}/{source.stem}")
            except Exception as error:
                reports.append({"lesson": source.stem, "kind": kind, "status": "failed", "error": str(error)})
                print(f"FAIL {kind}/{source.stem}: {error}", file=sys.stderr)
                break
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                manager.cleanup_resources()
    report_path = ROOT / "artifacts" / "notebook-validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"records": reports, "learner_mastery": "not_assessed"}, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(row["status"] == "failed" for row in reports):
        return 1
    print(f"Executed {len(reports)} notebooks. Learner mastery was not assessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
