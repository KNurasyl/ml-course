import math


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite real required; bool is not a measurement")
    return float(value)


def execute(request, release, cache):
    if not isinstance(request, dict) or set(request) != {"session", "turn", "x"}:
        raise ValueError("request schema")
    if not isinstance(request["session"], str) or not request["session"] or type(request["turn"]) is not int or request["turn"] < 0:
        raise ValueError("session and turn")
    x = number(request["x"])
    key = (request["session"], request["turn"])
    if key in cache:
        if cache[key]["x"] != x:
            raise ValueError("same operation, different input")
        return dict(cache[key]["response"])
    if not isinstance(release, dict) or set(release) != {"id", "mean", "scale", "weight", "bias"}:
        raise ValueError("release schema")
    if not isinstance(release["id"], str) or not release["id"]:
        raise ValueError("release id")
    mean, scale, w, b = (number(release[k]) for k in ("mean", "scale", "weight", "bias"))
    if scale <= 0:
        raise ValueError("positive scale required")
    prediction = w * ((x - mean) / scale) + b
    if not math.isfinite(prediction):
        raise ValueError("non-finite output")
    response = {"prediction": prediction, "release_id": release["id"]}
    cache[key] = {"x": x, "response": dict(response)}
    return response
