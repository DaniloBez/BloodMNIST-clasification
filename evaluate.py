import os
import json
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from config import Config

logger = logging.getLogger(__name__)

BLOODMNIST_CLASSES = [
    'Basophil', 'Eosinophil', 'Erythroblast', 'Immature Granulocyte',
    'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet'
]


def evaluate_pytorch_model(model, data_loader, device=Config.DEVICE):
    model.eval()
    model.to(device)

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.append(targets.view(-1).cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_probs = np.vstack(all_probs)

    return y_true, y_pred, y_probs


def compute_metrics(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')

    report_dict = classification_report(
        y_true, y_pred, target_names=BLOODMNIST_CLASSES, output_dict=True
    )

    weighted_avg = report_dict.get('weighted avg')
    weighted_f1 = float(weighted_avg['f1-score']) if isinstance(weighted_avg, dict) else 0.0
    macro_avg = report_dict.get('macro avg')
    macro_f1 = float(macro_avg['f1-score']) if isinstance(macro_avg, dict) else 0.0

    return {
        'accuracy': float(acc),
        'auc_roc': float(auc),
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'per_class_report': report_dict
    }


def plot_confusion_matrix(y_true, y_pred, save_path: str, model_name: str = "Model"):
    cm = confusion_matrix(y_true, y_pred)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=BLOODMNIST_CLASSES,
        yticklabels=BLOODMNIST_CLASSES,
        ax=ax
    )
    plt.title(f'Normalized Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_full_evaluation(y_true, y_pred, y_probs, model_name: str, config: Config = Config, meta: dict = None):
    logger.info(f"=== Оцінка якості для моделі: {model_name} ===")

    metrics = compute_metrics(y_true, y_pred, y_probs)
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"AUC-ROC:     {metrics['auc_roc']:.4f}")
    logger.info(f"F1 weighted: {metrics['weighted_f1']:.4f}")
    logger.info(f"F1 macro:    {metrics['macro_f1']:.4f}")

    logger.info(f"{'Клас':22s} {'precision':>9s} {'recall':>8s} {'f1':>8s} {'support':>8s}")
    for cls in BLOODMNIST_CLASSES:
        row = metrics['per_class_report'][cls]
        logger.info(
            f"{cls:22s} {row['precision']:9.3f} {row['recall']:8.3f} "
            f"{row['f1-score']:8.3f} {int(row['support']):8d}"
        )

    cm_path = os.path.join(config.RESULTS_DIR, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, cm_path, model_name=model_name)
    logger.info(f"Матрицю помилок збережено в: {cm_path}")

    if meta:
        metrics['meta'] = meta

    json_path = os.path.join(config.RESULTS_DIR, f'{model_name}_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    logger.info(f"Повний звіт збережено в: {json_path}")

    return metrics
