from pathlib import Path
import argparse, importlib.util, json
import torch
from torch import nn


def run_checks(module):
    torch.manual_seed(72)
    model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.8), nn.Linear(4, 2))
    model.train()
    x = torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]])
    y = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    actual = module.validation_loss(model, x, y, criterion)
    assert type(actual) is float, "Return the scalar value, not a tensor retaining a graph"
    assert model.training, "Restore previous train/eval state"
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items()), (
        "Validation changed parameters or buffers"
    )
    assert all(p.grad is None for p in model.parameters()), "Validation created gradients"
    model.eval()
    with torch.inference_mode():
        expected = float(criterion(model(x), y))
    assert abs(actual - expected) < 1e-6, "Dropout must be disabled in validation"
    second = module.validation_loss(model, x, y, criterion)
    assert not model.training and abs(second - actual) < 1e-6, "Repeated eval must preserve the previous mode"
    model.train()
    try:
        module.validation_loss(model, x, torch.tensor([0]), criterion)
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("Invalid target shape must not silently return a success value")
    assert model.training, "An exception must still restore the original mode"
    return {"checks": 8, "passed": True, "mastery": "requires your explanation and new data"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("submission", type=Path)
    a = p.parse_args()
    s = importlib.util.spec_from_file_location("student", a.submission)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    print(json.dumps(run_checks(m)))
