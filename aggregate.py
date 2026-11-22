import os
import re
import csv
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['Basophil', 'Eosinophil', 'Erythroblast', 'Immature Granulocyte',
           'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']
FILE_RE = re.compile(r'^(?P<group>.+)_s(?P<seed>\d+)_metrics\.json$')
GROUP_RE = re.compile(r'^(?P<base>[A-Z]+_\d+)(?:_(?P<tag>.+))?$')


def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*_metrics.json'))):
        m = FILE_RE.match(os.path.basename(path))
        if not m:
            return None

        with open(path, encoding='utf-8') as f:
            d = json.load(f)

        meta = d.get('meta') or {}
        training = meta.get('training') or {}
        report = d.get('per_class_report', {})
        macro_f1 = d.get('macro_f1')

        if macro_f1 is None and isinstance(report.get('macro avg'), dict):
            macro_f1 = report['macro avg']['f1-score']

        runs.append({
            'group': m['group'], 'seed': int(m['seed']),
            'accuracy': d['accuracy'], 'auc_roc': d['auc_roc'],
            'macro_f1': macro_f1, 'weighted_f1': d['weighted_f1'],
            'best_val_acc': training.get('best_val_acc'),
            'train_time_s': training.get('train_time_s'),
            'n_params': meta.get('n_params_total'),
            'per_class_f1': {c: report[c]['f1-score'] for c in CLASSES if c in report},
        })
    return runs


def stat(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    return mean, std, len(vals)


def fmt(mean, std, digits=4, scale=1.0):
    if mean is None:
        return '—'
    if std is None:
        return f"{mean * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def by_group(runs):
    groups = {}
    for r in runs:
        groups.setdefault(r['group'], []).append(r)
    return groups


def main_table(groups):
    lines = ["| Модель | Seed | Accuracy | AUC-ROC | F1 macro | F1 weighted | Параметри, млн | Час навчання, хв |",
             "|---|---|---|---|---|---|---|---|"]
    for name, rs in groups.items():
        row = [name, str(len(rs))]
        for key in ('accuracy', 'auc_roc', 'macro_f1', 'weighted_f1'):
            row.append(fmt(*stat([r[key] for r in rs])[:2]))
        params = [r['n_params'] for r in rs if r['n_params']]
        row.append(f"{params[0] / 1e6:.1f}" if params else '—')
        t_mean, t_std, n_t = stat([r['train_time_s'] for r in rs])
        row.append(fmt(t_mean, t_std, 1, 1 / 60) if n_t else '—')
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def per_class_table(groups):
    lines = ["| Модель | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    matrix = []
    for name, rs in groups.items():
        means = []
        for c in CLASSES:
            m, _, _ = stat([r['per_class_f1'].get(c) for r in rs])
            means.append(m)
        matrix.append(means)
        lines.append(f"| {name} | " + " | ".join('—' if m is None else f"{m:.3f}" for m in means) + " |")
    return "\n".join(lines), matrix


def ablation_table(main_groups, ab_groups):
    lines = ["| Конфігурація | Seed | Val acc | Test acc | Δ test acc до бази | База (ті самі seed) |",
             "|---|---|---|---|---|---|"]
    for name, rs in ab_groups.items():
        m = GROUP_RE.match(name)
        base_name = m['base']
        seeds = {r['seed'] for r in rs}
        base_rs = [r for r in main_groups.get(base_name, []) if r['seed'] in seeds]
        a_mean, a_std, _ = stat([r['accuracy'] for r in rs])
        v_mean, v_std, n_v = stat([r['best_val_acc'] for r in rs])
        b_mean, b_std, _ = stat([r['accuracy'] for r in base_rs])
        delta = '—' if (a_mean is None or b_mean is None) else f"{(a_mean - b_mean) * 100:+.2f} п.п."
        base_txt = f"{base_name}: {fmt(b_mean, b_std)} (n={len(base_rs)})" if base_rs else f"{base_name}: немає запусків із такими seed"
        lines.append(f"| {name} | {len(rs)} | {fmt(v_mean, v_std) if n_v else '—'} | {fmt(a_mean, a_std)} | {delta} | {base_txt} |")
    return "\n".join(lines)


def plot_accuracy(groups, path):
    names = list(groups)
    means = [stat([r['accuracy'] for r in groups[n]])[0] for n in names]
    stds = [stat([r['accuracy'] for r in groups[n]])[1] or 0.0 for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, means, yerr=stds, capsize=5, color='#4C78A8')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.002, f"{m:.3f}", ha='center', fontsize=9)
    ax.set_ylim(max(0.0, min(means) - 0.05), 1.0)
    ax.set_ylabel('Test accuracy (mean ± std по seed)')
    ax.set_title('BloodMNIST: порівняння моделей')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_per_class(groups, matrix, path):
    data = np.array([[np.nan if v is None else v for v in row] for row in matrix])
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(groups) + 2))
    im = ax.imshow(data, cmap='viridis', vmin=max(0.0, np.nanmin(data) - 0.05), vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha='right')
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(list(groups))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha='center', va='center', color='white', fontsize=8)
    ax.set_title('F1 по класах (середнє по seed)')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='./results')
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        print("Не знайдено файлів *_s<seed>_metrics.json у", args.results_dir)
        return
    groups = by_group(runs)
    main_groups = {n: groups[n] for n in sorted(groups) if not GROUP_RE.match(n)['tag']}
    ab_groups = {n: groups[n] for n in sorted(groups) if GROUP_RE.match(n)['tag']}

    out = ["# Зведені результати (середнє ± стандартне відхилення по seed)\n",
           "## Основне порівняння\n", main_table(main_groups), ""]
    if main_groups:
        pc_text, matrix = per_class_table(main_groups)
        out += ["## F1 по класах (середнє по seed)\n", pc_text, ""]
        plot_accuracy(main_groups, os.path.join(args.results_dir, 'summary_accuracy.png'))
        plot_per_class(main_groups, matrix, os.path.join(args.results_dir, 'summary_per_class_f1.png'))
    if ab_groups:
        out += ["## Абляції\n", ablation_table(main_groups, ab_groups), ""]

    with open(os.path.join(args.results_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(out))

    with open(os.path.join(args.results_dir, 'summary_runs.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['group', 'seed', 'accuracy', 'auc_roc', 'macro_f1', 'weighted_f1',
                    'best_val_acc', 'train_time_s', 'n_params'])
        for r in runs:
            w.writerow([r['group'], r['seed'], r['accuracy'], r['auc_roc'], r['macro_f1'],
                        r['weighted_f1'], r['best_val_acc'], r['train_time_s'], r['n_params']])
    print("\n".join(out))


if __name__ == '__main__':
    main()
