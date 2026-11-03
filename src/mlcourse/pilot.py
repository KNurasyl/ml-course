"""Small, inspectable CPU reference implementations for the pilot course.

All datasets are synthetic teaching examples, never evidence about candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
import json
import math
from numbers import Real

import numpy as np


def vector(values, name="values") -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name}: expected a non-empty one-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: all values must be finite")
    return result


def paired(actual, predicted):
    actual = vector(actual, "actual")
    predicted = vector(predicted, "predicted")
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have equal shapes")
    return actual, predicted


def mse(actual, predicted) -> float:
    actual, predicted = paired(actual, predicted)
    with np.errstate(over="raise", invalid="raise"):
        try:
            return float(np.mean(np.square(actual - predicted)))
        except FloatingPointError as error:
            raise ValueError("MSE overflow: rescale inputs") from error


def predict_line(x, w: float, b: float) -> np.ndarray:
    x = vector(x, "x")
    if not math.isfinite(w) or not math.isfinite(b):
        raise ValueError("parameters must be finite")
    result = w * x + b
    if not np.isfinite(result).all():
        raise ValueError("prediction overflow")
    return result


def linear_gradient(x, y, w, b):
    x, y = paired(x, y)
    residual = predict_line(x, w, b) - y
    return float(2 * np.mean(residual * x)), float(2 * np.mean(residual))


def central_difference(function, value, epsilon=1e-5):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    answer = (function(value + epsilon) - function(value - epsilon)) / (2 * epsilon)
    if not math.isfinite(answer):
        raise ValueError("non-finite finite difference")
    return answer


@dataclass(frozen=True)
class Standardizer:
    mean: float
    scale: float

    def transform(self, values):
        return (vector(values) - self.mean) / self.scale


def fit_standardizer(train_values):
    train_values = vector(train_values)
    deviation = float(train_values.std(ddof=0))
    return Standardizer(float(train_values.mean()), deviation if deviation else 1.0)


def matvec(matrix, weights, bias=0.0):
    matrix = np.asarray(matrix, dtype=float)
    weights = vector(weights, "weights")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != weights.size:
        raise ValueError("expected (rows, features) matrix and matching weights")
    if not np.isfinite(matrix).all() or not math.isfinite(bias):
        raise ValueError("matrix and bias must be finite")
    return matrix @ weights + bias


def fit_gradient(x, y, *, learning_rate=0.1, steps=500, w=0.0, b=0.0):
    x, y = paired(x, y)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    initial = mse(y, predict_line(x, w, b))
    history = [initial]
    limit = max(initial, 1.0) * 1e8
    for _ in range(steps):
        dw, db = linear_gradient(x, y, w, b)
        next_w, next_b = w - learning_rate * dw, b - learning_rate * db
        if not np.isfinite([next_w, next_b]).all():
            return {"w": w, "b": b, "history": history, "status": "diverged"}
        try:
            loss = mse(y, predict_line(x, next_w, next_b))
        except ValueError:
            return {"w": w, "b": b, "history": history, "status": "diverged"}
        if loss > limit:
            return {"w": w, "b": b, "history": history, "status": "diverged"}
        w, b = next_w, next_b
        history.append(loss)
    return {"w": w, "b": b, "history": history, "status": "completed"}


def make_dataset(seed=2026, n=100, noise=0.35):
    if not isinstance(n, int) or n < 5 or noise < 0 or not math.isfinite(noise):
        raise ValueError("n >= 5 and finite nonnegative noise required")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = 2 * x + 1 + rng.normal(0, noise, n)
    order = rng.permutation(n)
    train_end, valid_end = int(n * 0.6), int(n * 0.8)
    split = np.empty(n, dtype=object)
    split[order[:train_end]] = "train"
    split[order[train_end:valid_end]] = "validation"
    split[order[valid_end:]] = "test"
    return {"row_id": np.arange(n), "x": x, "y": y, "split": split}


def run_experiment(seed=2026, noise=0.35):
    data = make_dataset(seed=seed, noise=noise)
    x, y, split = data["x"], data["y"], data["split"]
    train = split == "train"
    validation = split == "validation"
    test = split == "test"
    baseline = float(y[train].mean())
    design = np.column_stack([x[train], np.ones(train.sum())])
    w, b = np.linalg.lstsq(design, y[train], rcond=None)[0]
    validation_scores = {
        "baseline": mse(y[validation], np.full(validation.sum(), baseline)),
        "linear": mse(y[validation], predict_line(x[validation], w, b)),
    }
    selected = min(validation_scores, key=validation_scores.get)
    prediction = (predict_line(x[test], w, b) if selected == "linear"
                  else np.full(test.sum(), baseline))
    return {
        "seed": seed, "noise": noise, "selected_on_validation": selected,
        "counts": {name: int(np.sum(split == name)) for name in ("train", "validation", "test")},
        "parameters": {"w": float(w), "b": float(b)},
        "validation_mse": validation_scores,
        "test_mse": mse(y[test], prediction),
        "test_baseline_mse": mse(y[test], np.full(test.sum(), baseline)),
        "baseline_mean": baseline,
    }


def predict_payload(payload):
    """Reference boundary for the tiny P05 service, not a general ML server."""
    if not isinstance(payload, dict) or set(payload) != {"x"}:
        raise ValueError("request must contain exactly one field: x")
    x = payload["x"]
    if isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x):
        raise ValueError("x must be a finite number")
    prediction = 2.0 * float(x) + 1.0
    if not math.isfinite(prediction):
        raise ValueError("prediction overflow")
    return {"prediction": prediction, "model_version": "v1"}


def prediction_handler():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("invalid body size")
                request = json.loads(self.rfile.read(size))
                reply = predict_payload(request)
                status = 200
            except (ValueError, TypeError, UnicodeDecodeError):
                status, reply = 400, {"error": "expected JSON with a finite numeric x"}
            body = json.dumps(reply, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    return Handler
