# BloodMNIST: порівняння моделей

Моделі: Random Forest, ResNet-18 (ImageNet), ViT-B/16, Swin-T.

## Запуск окремого експерименту
    python main.py --model rf --image_size 28
    python main.py --model cnn --image_size 224 --seed 101
    python main.py --model vit            # ViT/Swin завжди 224x224
    python main.py --model vit --seed 323 --lr 1e-4 --scheduler cosine --tag lr1e-4cos   # абляція

Опції: `--seed`, `--epochs`, `--lr`, `--scheduler {none,cosine}`, `--augmentation {basic,crop,color}`, `--tag`.

## Усі експерименти (з відновленням: готові пропускаються)
    python run_all.py --stage main        # основне порівняння (однакові 12 епох для всіх нейромереж)
    python run_all.py --stage ablation
    python run_all.py --stage all --dry_run   # лише показати план

## Зведення результатів
    python aggregate.py     # results/summary.md, summary_runs.csv, summary_accuracy.png, summary_per_class_f1.png

## Іменування файлів
`<МОДЕЛЬ>_<розмір>[_<тег>]_s<seed>_*` у `results/` (metrics.json, history.csv, графіки), логи в `results/logs/`.
