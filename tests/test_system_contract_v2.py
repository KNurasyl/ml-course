import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]


def load(path):
    s=importlib.util.spec_from_file_location('contract_'+Path(path).stem,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_reference_and_deliberately_broken_retry():
    check=load('exercises/M06/check.py').checks
    good=load('solutions/exercises/M06/solution.py').execute
    assert check(good)['passed']
    def broken(req,release,cache):
        return good(req,release,{})
    with pytest.raises(AssertionError,match='retry'):check(broken)
