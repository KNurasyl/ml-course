"""Meaningful deterministic boundary checks for M12-M16 course labs."""
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def labs():
    return {m: runpy.run_path(str(ROOT / f'labs/{m}/demo.py')) for m in ['M13', 'M14', 'M15', 'M16']}


def test_release_canary_detects_wrong_behavior(labs):
    stable = {'schema': 1, 'version': 'a', 'weight': 2, 'bias': 1}
    assert labs['M13']['canary'](stable)
    assert not labs['M13']['canary'](dict(stable, weight=20))


@pytest.mark.parametrize('bad', [True, float('nan'), float('inf'), '2', None])
def test_contract_rejects_non_finite_and_wrong_types(labs, bad):
    with pytest.raises(ValueError):
        labs['M13']['predict']({'schema': 1, 'version': 'a', 'weight': 2, 'bias': 1}, {'x': bad})


def test_agent_never_reads_other_session(labs):
    result = labs['M14']['execute']([{'tool': 'read_document', 'args': {'document_id': 'd2'}}])
    assert result['stop'] == 'not_available'
    assert result['observations'] == []
    assert result['trace'] == []


def test_wer_counts_insertions_and_empty_hypothesis(labs):
    wer = labs['M15']['wer']
    assert wer('a', 'a b c') == 2
    assert wer('a b', '') == 1
    assert wer('A, b!', 'a b') == 0
    with pytest.raises(ValueError):
        wer('', 'a')


def test_late_cancelled_audio_is_never_accepted(labs):
    accept = labs['M15']['accept_chunk']
    assert not accept(12, 11, {11})
    assert not accept(12, 12, {12})
    assert accept(12, 12, {11})


def test_hr_scores_cannot_be_used_for_technical_case(labs):
    case = {'track': 'technical', 'scores': {'clarity': 2, 'specificity': 1, 'reasoning': 1},
            'evidence': 'example', 'next_step': 'practice'}
    with pytest.raises(ValueError, match='rubric'):
        labs['M16']['validate_case'](case)


def test_lookup_hw_all_ten_expected_paths():
    assert runpy.run_path(str(ROOT / 'solutions/modules/M14_hw.py'))['run']()['checks'] == 10
