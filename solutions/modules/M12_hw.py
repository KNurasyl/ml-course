"""Reference HW: frozen bigram base, rank-one adaptation on mixed grammar."""
from pathlib import Path
import json
import runpy
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def run():
    initial = runpy.run_path(str(ROOT / 'labs/M12/demo.py'))['run']()
    ckpt = torch.load(ROOT / 'artifacts/M12/tiny-lm.pt', weights_only=True)
    torch.manual_seed(17)
    table = {c: i for i, c in enumerate(ckpt['alphabet'])}
    base = ckpt['base'].clone()

    def data(pattern, lengths):
        seqs = ['^' + pattern * n + '$' for n in lengths]
        return (torch.tensor([table[c] for s in seqs for c in s[:-1]]),
                torch.tensor([table[c] for s in seqs for c in s[1:]]))

    old_train, new_train = data('ab', range(2, 10)), data('ac', range(2, 10))
    train = tuple(torch.cat((old_train[i], new_train[i])) for i in (0, 1))
    old_test, new_test = data('ab', range(10, 14)), data('ac', range(10, 14))
    a = nn.Parameter(torch.randn(5, 1) * .05)
    b = nn.Parameter(torch.zeros(1, 5))
    optimizer = torch.optim.Adam([a, b], lr=.08)
    criterion = nn.CrossEntropyLoss()
    for _ in range(240):
        optimizer.zero_grad()
        criterion(base[train[0]] + a[train[0]] @ b, train[1]).backward()
        optimizer.step()
    with torch.no_grad():
        old = float(criterion(base[old_test[0]] + a[old_test[0]] @ b, old_test[1]))
        new = float(criterion(base[new_test[0]] + a[new_test[0]] @ b, new_test[1]))
    assert torch.equal(base, ckpt['base'])
    assert old < initial['adapted_old_loss']
    assert new < initial['base_new_loss']
    report = {'mixed_old_loss': old, 'mixed_new_loss': new,
              'single_task_old_loss': initial['adapted_old_loss'],
              'single_task_new_loss': initial['adapted_new_loss'],
              'limitation': 'a has two possible next tokens; a bigram model cannot distinguish the grammar'}
    (ROOT / 'artifacts/M12/mixed-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
