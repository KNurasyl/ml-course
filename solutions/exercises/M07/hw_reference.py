"""Complete M07-HW01 reference: changed data, plots, observed errors, finite difference."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/M07/hw-reference'


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, str(ROOT / 'labs/M07/demo.py'), '--seed', '23',
         '--noise', '0.30', '--output-dir', str(OUT)],
        capture_output=True, text=True, check=True,
    )
    (OUT / 'training.log').write_text(completed.stdout, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('m07_reference_demo', ROOT / 'labs/M07/demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    config = json.loads((OUT / 'release.json').read_text(encoding='utf-8'))
    split = json.loads((OUT / 'split.json').read_text(encoding='utf-8'))
    history = json.loads((OUT / 'history.json').read_text(encoding='utf-8'))
    training_report = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
    x, y = make_moons(n_samples=1000, noise=config['noise'], random_state=config['seed'])
    mean, scale = np.array(config['mean']), np.array(config['scale'])
    transformed = (x - mean) / scale
    train, valid, test = (np.array(split[key]) for key in ['train', 'validation', 'test'])
    assert np.allclose(mean, x[train].mean(axis=0))
    assert not set(train) & set(valid) and not set(train) & set(test) and not set(valid) & set(test)
    model = demo.network()
    model.load_state_dict(torch.load(OUT / 'model.pt', map_location='cpu', weights_only=True))
    model.eval()
    with torch.inference_mode():
        logits = model(torch.tensor(transformed[test], dtype=torch.float32))
        probabilities = logits.softmax(dim=-1).numpy()
    predictions = probabilities.argmax(axis=1)
    wrong_positions = np.flatnonzero(predictions != y[test])
    assert len(wrong_positions) >= 2, 'The fixed reference must contain two real errors.'
    # Select the first two by original dataset ID; do not tune the model using these test errors.
    selected = sorted(wrong_positions.tolist(), key=lambda p: int(test[p]))[:2]
    errors = []
    for position in selected:
        index = int(test[position])
        distances = np.linalg.norm(transformed[train] - transformed[index], axis=1)
        near = train[np.argsort(distances)[:15]]
        counts = np.bincount(y[near], minlength=2)
        errors.append({
            'dataset_index': index, 'test_position': position,
            'raw_features': x[index].tolist(), 'scaled_features': transformed[index].tolist(),
            'target': int(y[index]), 'predicted': int(predictions[position]),
            'logits': logits[position].tolist(), 'probabilities': probabilities[position].tolist(),
            'nearest_15_train_indices': near.tolist(), 'nearest_15_train_class_counts': counts.tolist(),
            'interpretation': (
                'Observed misclassification. Neighbour counts describe this fitted feature space; '
                'they do not identify whether noise, limited capacity or sampling caused this error.'
            ),
        })

    # Check the derivative against central finite differences in float64 arithmetic.
    def loss(w):
        return (w * 2.0 + 1.0 - 5.0) ** 2
    epsilon = 1e-5
    finite_difference = (loss(1.0 + epsilon) - loss(1.0 - epsilon)) / (2 * epsilon)
    analytic = -8.0
    assert abs(finite_difference - analytic) < 1e-8
    selected_epoch = int(np.argmin([row['val_loss'] for row in history])) + 1
    assert selected_epoch == training_report['best_epoch']
    assert abs(float((predictions == y[test]).mean()) - training_report['mlp_test_accuracy']) < 1e-6
    assert training_report['fresh_process']['fresh_process_max_abs_error'] < 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot([row['epoch'] for row in history], [row['train_loss'] for row in history], label='Обучение (train)')
    axes[0].plot([row['epoch'] for row in history], [row['val_loss'] for row in history], label='Проверка (validation)')
    axes[0].axvline(selected_epoch, color='black', linestyle='--', label=f'Выбрана эпоха {selected_epoch}')
    axes[0].set(xlabel='Эпоха', ylabel='Перекрёстная энтропия', title='Выбор checkpoint без доступа к test')
    axes[0].legend()
    for cls, color in [(0, '#2667bd'), (1, '#e38b2c')]:
        mask = y[test] == cls
        axes[1].scatter(x[test[mask], 0], x[test[mask], 1], s=20, c=color, alpha=.7, label=f'Истинный класс {cls}')
    wrong_ids = test[wrong_positions]
    axes[1].scatter(x[wrong_ids, 0], x[wrong_ids, 1], s=90, facecolors='none', edgecolors='crimson', label='Ошибка предсказания')
    for error in errors:
        point = x[error['dataset_index']]
        axes[1].annotate(f"id={error['dataset_index']}", point, xytext=(15, 22), textcoords='offset points',
                         fontsize=10, arrowprops={'arrowstyle': '->'}, bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
    axes[1].set(xlabel='Исходный признак x1', ylabel='Исходный признак x2', title='Классы test и реальные ошибки')
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / 'loss-and-errors.png', dpi=160)
    plt.close(fig)

    report = {
        'seed': 23, 'noise': .30, 'training_report': training_report,
        'test_error_count': int(len(wrong_positions)), 'errors': errors,
        'finite_difference': finite_difference, 'analytic_gradient': analytic,
        'finite_difference_epsilon': epsilon,
        'plots': ['loss-and-errors.png'],
        'selection_policy': 'first two mistakes sorted by original dataset index, after frozen checkpoint selection',
        'conclusion': 'Metrics describe this split. Test error analysis did not select or change the checkpoint.',
    }
    (OUT / 'hw-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [
        '# M07-HW01 — исполненный эталон', '',
        'Данные: seed=23, noise=0,30; split 600/200/200. StandardScaler обучен только на train.', '',
        '| Метод | Test accuracy |', '|---|---:|',
        f"| MLP | {training_report['mlp_test_accuracy']:.6f} |",
        f"| LogisticRegression | {training_report['logistic_test_accuracy']:.6f} |", '',
        f"Checkpoint выбран по минимуму validation: эпоха {selected_epoch}. Test не использовался для выбора.", '',
        '![Loss и реальные ошибки](loss-and-errors.png)', '',
        '## Две ошибки', '',
    ]
    for error in errors:
        probs = error['probabilities']
        lines += [
            f"### Объект {error['dataset_index']}", '',
            f"Координаты {error['raw_features']}; истинный класс {error['target']}, предсказан {error['predicted']}.",
            f"Logits {error['logits']}; вероятности классов 0/1: {probs[0]:.6f}/{probs[1]:.6f}.",
            f"Среди 15 ближайших train-объектов в стандартизованном пространстве: {error['nearest_15_train_class_counts']} объектов классов 0/1.",
            'Это конкретное несогласие модели с размеченной целью. Окружение точки и вероятность помогают описать ошибку, но не доказывают её причину. '
            'Нельзя по координатам установить, вызван ли промах шумом, ограниченной моделью или выборкой. Для различения гипотез нужны отдельные эксперименты на validation; этот test не превращаем в подбор параметров.', '',
        ]
    lines += [
        '## Проверки', '',
        f'Производная по w аналитически −8; центральная разность с ε={epsilon}: {finite_difference:.10f}.',
        f"Максимальная разница logits при восстановлении в новом процессе: {training_report['fresh_process']['fresh_process_max_abs_error']}.",
        'Артефакты model.pt, release.json, split.json, history.json и report.json находятся рядом с этим отчётом. '
        'Вероятности показывают уверенность выбранной модели, а не вероятность истинной причины ошибки.', '',
    ]
    (OUT / 'hw-report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'report': str(OUT / 'hw-report.json'), 'selected_error_ids': [e['dataset_index'] for e in errors],
                      'test_error_count': len(wrong_positions), 'gradient': finite_difference,
                      'accuracy': training_report['mlp_test_accuracy']}, indent=2))


if __name__ == '__main__':
    main()
