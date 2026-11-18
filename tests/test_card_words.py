"""R09: table cells, including the first column of the lesson card, keep whole words.

`.material` breaks long tokens anywhere, which also shrinks the min-content width of table cells, so the
automatic table layout squeezed the card's first column until words split ("Прав|ило"). Cells override it
with break-word: a word fits whole into the column's minimum width, an overlong token still wraps.
The browser check (472 first-column cells, 0 split words at 1280 and 375 px, both themes) is in the R09
journal entry; this test keeps the rule in place.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r09_reader", ROOT / "tools/build_reader.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def test_table_cells_override_anywhere_wrapping():
    css = reader.CSS
    assert re.search(r"\.material\{[^}]*overflow-wrap:anywhere", css)
    rules = [m for m in re.finditer(r"\.body-content th,\.body-content td\{([^}]*)\}", css)
             if "overflow-wrap:break-word" in m.group(1)]
    assert len(rules) == 1
    # The cell rule comes after .material, so in the cascade it wins for th and td.
    assert rules[0].start() > css.index(".material{")
    assert rules[0].group(0) in (ROOT / "START_HERE.html").read_text(encoding="utf-8")
