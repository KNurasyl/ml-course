"""Execute pilot notebooks in fresh kernels without changing source notebooks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parents[1]


def rebase_markdown_links(markdown: str, source_dir: Path, output_dir: Path) -> str:
    """Keep inline and reference links pointing at their original local targets."""
    def rebase_target(target: str) -> str:
        angled = target.startswith("<") and target.endswith(">")
        value = target[1:-1] if angled else target
        url = urlsplit(value)
        if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
            return target
        resolved = (source_dir / unquote(url.path)).resolve()
        relative = Path(os.path.relpath(resolved, output_dir)).as_posix()
        rewritten = urlunsplit(("", "", quote(relative, safe="/:@-._~!$&'()*+,;="), url.query, url.fragment))
        return f"<{rewritten}>" if angled else rewritten

    # Keep fenced and inline code examples verbatim. Links in code are examples,
    # not navigational targets, and executable notebook cells never enter here.
    segments = re.split(r"(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)", markdown)
    inline = re.compile(r"(?P<lead>!?\[[^\]\n]*\]\()(?P<target><[^>\n]+>|(?:[^\s()]|\([^()\n]*\))+)(?P<tail>(?:\s+\"[^\"]*\"|\s+'[^']*')?\s*\))")
    reference = re.compile(r"(?m)^(?P<lead> {0,3}\[[^\]\n]+\]:\s*)(?P<target><[^>\n]+>|\S+)")
    for index in range(0, len(segments), 2):
        segments[index] = inline.sub(lambda m: m["lead"] + rebase_target(m["target"]) + m["tail"], segments[index])
        segments[index] = reference.sub(lambda m: m["lead"] + rebase_target(m["target"]), segments[index])
    return "".join(segments)


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
                for cell in notebook.cells:
                    if cell.cell_type == "markdown":
                        cell.source = rebase_markdown_links(cell.source, source.parent, destination.parent)
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
