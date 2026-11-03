"""Exports must retain navigation when their directory is deeper than the source."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("notebook_runner", ROOT / "tools/run_notebooks.py")
runner = module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("source_folder, output_folder", [
    ("notebooks/pilot", "artifacts/executed/starter"),
    ("solutions/pilot", "artifacts/executed/solutions"),
])
def test_exports_preserve_local_targets_without_rewriting_urls_or_code(tmp_path, source_folder, output_folder):
    markdown = (
        "[Урок](../../course/pilot/P01.md#practice)\n"
        "[Файл](<../../course/a%20b.md?mode=read#task> \"title\")\n"
        "[Источник](https://example.com/guide?q=1#part)\n"
        "[Якорь](#practice) [Запрос](?view=full)\n"
        "[ref]: ../../course/diagnostic.md#questions\n"
        "`[literal](../../course/pilot/P01.md)`\n"
        "```python\ntext = '[literal](../../course/pilot/P01.md)'\n```\n"
    )
    rebased = runner.rebase_markdown_links(markdown, tmp_path / source_folder, tmp_path / output_folder)
    assert "[Урок](../../../course/pilot/P01.md#practice)" in rebased
    assert "[Файл](<../../../course/a%20b.md?mode=read#task> \"title\")" in rebased
    assert "[ref]: ../../../course/diagnostic.md#questions" in rebased
    for unchanged in [
        "[Источник](https://example.com/guide?q=1#part)",
        "[Якорь](#practice) [Запрос](?view=full)",
        "`[literal](../../course/pilot/P01.md)`",
        "```python\ntext = '[literal](../../course/pilot/P01.md)'\n```",
    ]:
        assert unchanged in rebased
