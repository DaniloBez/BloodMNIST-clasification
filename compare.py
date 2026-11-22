"""Парні порівняння моделей на спільному тесті за збереженими передбаченнями (results/*_predictions.npz).

Двоступеневий bootstrap: перевибираємо зображення тесту (одні й ті самі для обох моделей, тому
враховується кореляція помилок) та seed (окремо для кожної моделі). Інтервал показує сумарну
невизначеність: через скінченний тест і через випадковість навчання.

Запуск: python compare.py [--results_dir ./results] [--n_boot 2000]
Створює results/compare.md. Використовує лише основні запуски (без тегів абляцій). Не залежить від torch.
"""
import os
import re
import glob
import argparse
import itertools
import numpy as np

FILE_RE = re.compile(r'^(?P<group>[A-Z]+_\d+)_s(?P<seed>\d+)_predictions\.npz$')


def load(results_dir):
    groups, y_true = {}, None
    for path in sorted(glob.glob(os.path.join(results_dir, '*_predictions.npz'))):
        m = FILE_RE.match(os.path.basename(path))
        if not m:
            continue
        d = np.load(path)
        if y_true is None:
            y_true = d['y_true']
        elif not np.array_equal(y_true, d['y_true']):
            raise SystemExit(f"y_true у {path} відрізняється від інших запусків")
        groups.setdefault(m['group'], []).append((d['y_pred'] == y_true).astype(np.float32))
    return {g: np.stack(v) for g, v in sorted(groups.items())}, y_true


def bootstrap(correct_a, correct_b, n_boot, rng):
    """correct_*: (n_seeds, n_images) з 1.0 там, де модель вгадала. Якщо correct_b=None, повертає accuracy A."""
    n_img = correct_a.shape[1]
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_img, n_img)
        acc_a = correct_a[:, idx].mean(axis=1)
        a = acc_a[rng.integers(0, len(acc_a), len(acc_a))].mean()
        if correct_b is None:
            out[i] = a
        else:
            acc_b = correct_b[:, idx].mean(axis=1)
            out[i] = a - acc_b[rng.integers(0, len(acc_b), len(acc_b))].mean()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--n_boot', type=int, default=2000)
    args = parser.parse_args()

    groups, y_true = load(args.results_dir)
    if len(groups) < 1:
        print("Не знайдено файлів *_s<seed>_predictions.npz у", args.results_dir)
        return
    rng = np.random.default_rng(0)
    print(f"Зображень у тесті: {len(y_true)}, моделей: {len(groups)}, bootstrap: {args.n_boot}")

    lines = ["# Порівняння моделей на спільному тесті (bootstrap по зображеннях і seed)\n",
             "## Accuracy: середнє по seed і 95 % інтервал\n",
             "| Модель | Seed | Accuracy | 95 % ДІ |", "|---|---|---|---|"]
    for name, c in groups.items():
        d = bootstrap(c, None, args.n_boot, rng)
        lo, hi = np.percentile(d, [2.5, 97.5])
        lines.append(f"| {name} | {c.shape[0]} | {c.mean():.4f} | [{lo:.4f}; {hi:.4f}] |")

    lines += ["", "## Парні різниці (A − B), у відсоткових пунктах\n",
              "| A | B | Δ accuracy, п.п. | 95 % ДІ, п.п. | Висновок |", "|---|---|---|---|---|"]
    for a, b in itertools.combinations(groups, 2):
        d = bootstrap(groups[a], groups[b], args.n_boot, rng) * 100
        obs = (groups[a].mean() - groups[b].mean()) * 100
        lo, hi = np.percentile(d, [2.5, 97.5])
        verdict = "різниця значуща" if (lo > 0 or hi < 0) else "не відрізнити"
        lines.append(f"| {a} | {b} | {obs:+.2f} | [{lo:+.2f}; {hi:+.2f}] | {verdict} |")

    text = "\n".join(lines)
    with open(os.path.join(args.results_dir, 'compare.md'), 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    main()
