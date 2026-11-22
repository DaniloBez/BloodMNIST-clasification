import os
import gc
import time
import logging
import argparse
import numpy as np
import torch
from config import Config
from utils import set_seed, setup_logging, count_parameters, get_environment_info
from dataset import get_dataloaders
from evaluate import run_full_evaluation, BLOODMNIST_CLASSES
from models.random_forest import RandomForestModel
from models.cnn import CNNModel
from models.transformer import TransformerModel

logger = logging.getLogger(__name__)


def build_name(model_type: str, image_size: int, seed: int, tag: str = '') -> str:
    """Ім'я експерименту: ТИП_РОЗМІР[_тег]_s<seed>, напр. VIT_224_s323 або CNN_224_crop_s101."""
    return f"{model_type.upper()}_{image_size}" + (f"_{tag}" if tag else "") + f"_s{seed}"


def get_model(model_type: str, model_name: str):
    if model_type == 'rf':
        return RandomForestModel(name=model_name)
    elif model_type == 'cnn':
        return CNNModel(name=model_name)
    elif model_type in ('vit', 'swin'):
        return TransformerModel(name=model_name, variant=model_type)
    else:
        raise ValueError(f"Невідомий тип моделі: {model_type}")


def log_dataset_info(train_loader, val_loader, test_loader):
    logger.info(
        f"Розмір вибірок: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )
    for split_name, loader in (('train', train_loader), ('test', test_loader)):
        labels = np.asarray(loader.dataset.labels).flatten()
        counts = np.bincount(labels, minlength=Config.NUM_CLASSES)
        distribution = ", ".join(f"{name}={c}" for name, c in zip(BLOODMNIST_CLASSES, counts))
        logger.info(f"Розподіл класів ({split_name}): {distribution}")


def collect_hyperparameters(model_type: str, image_size: int) -> dict:
    params = {
        'image_size': image_size,
        'seed': Config.SEED,
        'augmentation': 'none' if model_type == 'rf' else Config.AUGMENTATION,
    }
    if model_type == 'rf':
        params.update({
            'n_estimators': Config.RF_N_ESTIMATORS,
            'max_depth': Config.RF_MAX_DEPTH,
        })
    else:
        params.update({
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.NUM_EPOCHS,
            'optimizer': 'AdamW',
            'learning_rate': Config.LEARNING_RATE,
            'scheduler': Config.SCHEDULER,
            'warmup_epochs': Config.WARMUP_EPOCHS if Config.SCHEDULER == 'cosine' else 0,
            'weight_decay': Config.WEIGHT_DECAY,
            'pretrained': 'ImageNet',
        })
    return params


def run_experiment(model_type: str, image_size: int, seed: int = None, tag: str = '',
                   lr: float = None, epochs: int = None, scheduler: str = None,
                   augmentation: str = None):
    Config.configure(seed=seed, epochs=epochs, lr=lr, scheduler=scheduler, augmentation=augmentation)
    set_seed(Config.SEED)
    Config.setup(image_size=image_size)

    model_name = build_name(model_type, image_size, Config.SEED, tag)
    log_path = os.path.join(Config.RESULTS_DIR, 'logs', f'{model_name}.log')
    setup_logging(log_path)

    logger.info(f"ЗАПУСК ЕКСПЕРИМЕНТУ: {model_name} | Size: {image_size}x{image_size} | seed {Config.SEED}")
    environment = get_environment_info()
    logger.info(f"Середовище: {environment}")
    hyperparameters = collect_hyperparameters(model_type, image_size)
    logger.info(f"Гіперпараметри: {hyperparameters}")

    train_loader, val_loader, test_loader = get_dataloaders(Config, augment=(model_type != 'rf'))
    log_dataset_info(train_loader, val_loader, test_loader)

    model = get_model(model_type, model_name)
    if hasattr(model, 'net'):
        n_total, n_trainable = count_parameters(model.net)
        logger.info(f"Параметрів: {n_total:,} (навчальних: {n_trainable:,})")
    else:
        n_total, n_trainable = None, None

    model.fit(train_loader, val_loader)

    logger.info("Оцінка на тестовій вибірці")
    start_time = time.time()
    y_true, y_pred, y_probs = model.predict(test_loader)
    inference_time = time.time() - start_time

    # зберігаємо передбачення, щоб згодом можна було порівнювати моделі парними тестами
    np.savez_compressed(
        os.path.join(Config.RESULTS_DIR, f"{model_name}_predictions.npz"),
        y_true=np.asarray(y_true).reshape(-1),
        y_pred=np.asarray(y_pred).reshape(-1),
        y_probs=np.asarray(y_probs, dtype=np.float32),
    )
    logger.info(
        f"Інференс: {inference_time:.1f} с на {len(y_true)} зображень "
        f"({inference_time / len(y_true) * 1000:.2f} мс/зображення, включно із завантаженням даних)"
    )

    meta = {
        'model': model.name,
        'tag': tag,
        'seed': Config.SEED,
        'hyperparameters': hyperparameters,
        'n_params_total': n_total,
        'n_params_trainable': n_trainable,
        'training': model.train_info,
        'inference_time_s': round(inference_time, 2),
        'n_test_images': int(len(y_true)),
        'environment': environment,
    }

    metrics = run_full_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        model_name=model.name,
        config=Config,
        meta=meta
    )

    # звільняємо воркери DataLoader, модель і пам'ять GPU перед наступним експериментом
    del model, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="BloodMNIST Classification Pipeline")

    parser.add_argument(
        '--model',
        type=str,
        choices=['rf', 'cnn', 'vit', 'swin', 'all'],
        default='cnn',
        help="Оберіть модель для навчання: 'rf', 'cnn', 'vit' (ViT-B/16), 'swin' (Swin-T) або 'all' для всіх експериментів"
    )

    parser.add_argument(
        '--image_size',
        type=int,
        choices=[28, 224],
        default=28,
        help="Розмір вхідного зображення: 28 або 224"
    )

    parser.add_argument('--seed', type=int, default=None, help="Seed (за замовчуванням з config.py)")
    parser.add_argument('--tag', type=str, default='', help="Мітка експерименту (для абляцій), потрапляє в імена файлів")
    parser.add_argument('--lr', type=float, default=None, help="Перевизначити learning rate")
    parser.add_argument('--epochs', type=int, default=None, help="Перевизначити кількість епох")
    parser.add_argument('--scheduler', type=str, choices=['none', 'cosine'], default=None,
                        help="'cosine' = warmup + косинусне спадання lr")
    parser.add_argument('--augmentation', type=str, choices=['basic', 'crop', 'color'], default=None,
                        help="Набір аугментацій")

    args = parser.parse_args()
    overrides = dict(seed=args.seed, tag=args.tag, lr=args.lr, epochs=args.epochs,
                     scheduler=args.scheduler, augmentation=args.augmentation)

    if args.model == 'all':
        experiments = [
            ('rf', 28),
            ('cnn', 28),
            ('cnn', 224),
            ('vit', 224),
            ('swin', 224),
        ]

        results = {}
        for m_type, size in experiments:
            results[f"{m_type}_{size}"] = run_experiment(m_type, size, **overrides)

        logger.info("     ПІДСУМКОВІ РЕЗУЛЬТАТИ ВСІХ МОДЕЛЕЙ")
        for name, res in results.items():
            train_time = res['meta']['training'].get('train_time_s', 0)
            logger.info(
                f"{name:12s} | Acc: {res['accuracy']:.4f} | AUC: {res['auc_roc']:.4f} | "
                f"F1 w: {res['weighted_f1']:.4f} | F1 macro: {res['macro_f1']:.4f} | "
                f"Час навчання: {train_time / 60:.1f} хв"
            )

    else:
        image_size = args.image_size
        if args.model in ('vit', 'swin') and image_size != 224:
            print("ViT та Swin працюють лише з 224x224, встановлюю image_size=224")
            image_size = 224
        run_experiment(args.model, image_size, **overrides)


if __name__ == '__main__':
    main()