"""Послідовний запуск експериментів з підтримкою відновлення.

Етапи (--stage):
    main      основне порівняння: однакові гіперпараметри для всіх, 5 seed (RF, ResNet-18 x2, Swin, ViT)
    ablation  аугментації (crop, color) і 25 епох для ResNet-18-224, по 3 seed
    tune      підбір lr для ResNet-18-224, Swin, ViT: lr x {3e-5, 1e-4, 3e-4}, cosine, 1 seed
    tuned     фінал із підібраним lr: 5 seed, cosine (потрібен --tuned_lr)
    all       main + ablation + tune

Приклади:
    python run_all.py --stage main --dry_run
    python run_all.py --stage main
    python run_all.py --stage tune
    python run_all.py --stage tuned --tuned_lr cnn=1e-4 swin=1e-4 vit=3e-4

Seed-мажорний порядок: спочатку всі моделі для першого seed, потім для другого тощо,
тож при дострокової зупинці лишаються повні набори seed.
"""
import os
import argparse
import logging
import traceback

from config import Config
from main import run_experiment, build_name

SEEDS = [323, 101, 202, 303, 404]

MAIN_PLAN = [
    ('rf', 28, 5),
    ('cnn', 28, 5),
    ('cnn', 224, 5),
    ('swin', 224, 5),
    ('vit', 224, 5),
]

ABLATIONS = [
    dict(model='cnn', size=224, tag='crop', n_seeds=3, augmentation='crop'),
    dict(model='cnn', size=224, tag='color', n_seeds=3, augmentation='color'),
    dict(model='cnn', size=224, tag='ep25', n_seeds=3, epochs=25),
]

TUNE_MODELS = [('cnn', 224), ('swin', 224), ('vit', 224)]
TUNE_LRS = [('3e-5', 3e-5), ('1e-4', 1e-4), ('3e-4', 3e-4)]

TUNED_N_SEEDS = 5


def build_jobs(stage: str, tuned_lrs: dict = None):
    jobs = []
    if stage in ('main', 'all'):
        for i, seed in enumerate(SEEDS):
            for model, size, n_seeds in MAIN_PLAN:
                if i < n_seeds:
                    jobs.append(dict(model=model, size=size, seed=seed, tag=''))
    if stage in ('ablation', 'all'):
        for i, seed in enumerate(SEEDS):
            for ab in ABLATIONS:
                if i < ab['n_seeds']:
                    job = {k: v for k, v in ab.items() if k != 'n_seeds'}
                    job['seed'] = seed
                    jobs.append(job)
    if stage in ('tune', 'all'):
        for model, size in TUNE_MODELS:
            for label, lr in TUNE_LRS:
                jobs.append(dict(model=model, size=size, seed=SEEDS[0], tag=f"lr{label}cos",
                                 lr=lr, scheduler='cosine'))
    if stage == 'tuned':
        for seed in SEEDS[:TUNED_N_SEEDS]:
            for model, size in TUNE_MODELS:
                if model in (tuned_lrs or {}):
                    jobs.append(dict(model=model, size=size, seed=seed, tag='tuned',
                                     lr=tuned_lrs[model], scheduler='cosine'))
    return jobs


def parse_tuned_lrs(pairs):
    result = {}
    for pair in pairs or []:
        name, _, value = pair.partition('=')
        if name not in {m for m, _ in TUNE_MODELS} or not value:
            raise SystemExit(f"Невірний --tuned_lr '{pair}'. Формат: cnn=1e-4 swin=1e-4 vit=3e-4")
        result[name] = float(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['main', 'ablation', 'tune', 'tuned', 'all'], default='main')
    parser.add_argument('--tuned_lr', nargs='+', default=None,
                        help="Для --stage tuned: підібраний lr для кожної моделі, напр. cnn=1e-4 swin=1e-4 vit=3e-4")
    parser.add_argument('--dry_run', action='store_true', help="Лише показати план")
    args = parser.parse_args()

    tuned_lrs = parse_tuned_lrs(args.tuned_lr)
    if args.stage == 'tuned' and not tuned_lrs:
        raise SystemExit("Для --stage tuned вкажіть --tuned_lr, напр.: --tuned_lr cnn=1e-4 swin=1e-4 vit=3e-4")

    jobs = build_jobs(args.stage, tuned_lrs)
    failed = []
    for n, job in enumerate(jobs, 1):
        name = build_name(job['model'], job['size'], job['seed'], job.get('tag', ''))
        done = os.path.exists(os.path.join(Config.RESULTS_DIR, f"{name}_metrics.json"))
        print(f"[{n}/{len(jobs)}] {name}: {'пропуск (вже є)' if done else 'запуск'}")
        if done or args.dry_run:
            continue

        kwargs = dict(seed=job['seed'], tag=job.get('tag', ''), lr=job.get('lr'),
                      epochs=job.get('epochs'), scheduler=job.get('scheduler'),
                      augmentation=job.get('augmentation'))
        try:
            run_experiment(job['model'], job['size'], **kwargs)
        except Exception:
            failed.append(name)
            logging.getLogger().error(f"Експеримент {name} завершився помилкою:\n{traceback.format_exc()}")

    if failed:
        print("Не вдалося виконати:", ", ".join(failed))
    print("Готово. Для зведення результатів: python aggregate.py")


if __name__ == '__main__':
    main()