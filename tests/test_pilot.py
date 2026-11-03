import json
import threading
import subprocess
import sys
from pathlib import Path
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from mlcourse.pilot import (
    central_difference, fit_gradient, fit_standardizer, linear_gradient,
    make_dataset, matvec, mse, predict_line, predict_payload,
    prediction_handler, run_experiment,
)


def test_mse_known_value():
    assert mse([3, 5, 7], [2, 3, 4]) == pytest.approx(14 / 3)


@pytest.mark.parametrize("a,b", [([], []), ([1], [1, 2]), ([float("nan")], [1]),
                                 ([1], [float("inf")]), ([[1]], [[1]])])
def test_mse_rejects_invalid_shape_or_values(a, b):
    with pytest.raises(ValueError):
        mse(a, b)


def test_gradient_matches_finite_difference():
    x, y = [0, 1, 2], [1, 3, 5]
    dw, db = linear_gradient(x, y, 0, 0)
    assert (dw, db) == pytest.approx((-26 / 3, -6))
    assert dw == pytest.approx(central_difference(lambda w: mse(y, predict_line(x, w, 0)), 0))
    assert db == pytest.approx(central_difference(lambda b: mse(y, predict_line(x, 0, b)), 0))


def test_small_step_trains_and_large_step_stops():
    trained = fit_gradient([0, 1, 2], [1, 3, 5])
    assert trained["status"] == "completed"
    assert trained["history"][-1] < 1e-10
    assert [trained["w"], trained["b"]] == pytest.approx([2, 1], abs=1e-4)
    divergent = fit_gradient([0, 1, 2], [1, 3, 5], learning_rate=10)
    assert divergent["status"] == "diverged"
    assert len(divergent["history"]) < 501
    assert np.isfinite(divergent["history"]).all()


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_learning_rate(rate):
    with pytest.raises(ValueError):
        fit_gradient([0, 1], [1, 3], learning_rate=rate)


def test_scaler_uses_training_only():
    scaler = fit_standardizer([0, 1, 2])
    assert scaler.mean == 1
    assert scaler.scale == pytest.approx(np.sqrt(2 / 3))
    np.testing.assert_allclose(scaler.transform([3, 4]), np.array([2, 3]) / scaler.scale)
    assert fit_standardizer([0, 1, 2, 3, 4]).mean != scaler.mean
    np.testing.assert_allclose(fit_standardizer([2, 2]).transform([2, 3]), [0, 1])


def test_matrix_shapes_and_values():
    np.testing.assert_allclose(matvec([[1, 2, 3], [4, 5, 6]], [1, 0, -1]), [-2, -2])
    with pytest.raises(ValueError):
        matvec([[1, 2]], [1, 2, 3])


def test_split_reproducibility_and_disjointness():
    first, second = make_dataset(), make_dataset()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    ids = {name: set(first["row_id"][first["split"] == name]) for name in ("train", "validation", "test")}
    assert list(map(len, ids.values())) == [60, 20, 20]
    assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"])
    assert len(set.union(*ids.values())) == 100


def test_experiment_validation_selection_and_holdout():
    result = run_experiment()
    assert result == run_experiment()
    assert result["selected_on_validation"] == min(result["validation_mse"], key=result["validation_mse"].get)
    assert result["test_mse"] < result["test_baseline_mse"]


@pytest.mark.parametrize("payload", [{}, {"x": "4"}, {"x": True}, {"x": float("nan")}, {"x": 1, "extra": 2}, []])
def test_request_validation(payload):
    with pytest.raises(ValueError):
        predict_payload(payload)


def test_http_predict_and_bad_request():
    server = HTTPServer(("127.0.0.1", 0), prediction_handler())
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/predict"
    try:
        with urlopen(Request(endpoint, data=b'{"x":4}', headers={"Content-Type": "application/json"}), timeout=5) as response:
            assert json.load(response) == {"prediction": 9.0, "model_version": "v1"}
        with pytest.raises(HTTPError) as error:
            urlopen(Request(endpoint, data=b'{"x":"bad"}'), timeout=5)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_submission_checker_rejects_signed_mean(tmp_path):
    submission = tmp_path / "wrong.py"
    submission.write_text("def mse(a, b):\n    return sum(x-y for x,y in zip(a,b))/len(a)\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "tools" / "check_submission.py"
    result = subprocess.run([sys.executable, str(script), str(submission), "--task", "mse"], capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["learner_mastery"] == "not_assessed"
    assert report["behavior_passed"] is False


def test_submission_checker_accepts_reference_without_claiming_mastery(tmp_path):
    submission = tmp_path / "right.py"
    submission.write_text("from mlcourse.pilot import mse\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "tools" / "check_submission.py"
    result = subprocess.run([sys.executable, str(script), str(submission), "--task", "mse"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["behavior_passed"] is True
    assert report["learner_mastery"] == "not_assessed"
