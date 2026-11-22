import os
import sys
import csv
import time
import random
import logging
import platform
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 323):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_rf_data(data_loader):
    images_list = []
    labels_list = []

    for images, labels in data_loader:
        flattened_imgs = images.view(images.size(0), -1).numpy()
        images_list.append(flattened_imgs)
        labels_list.append(labels.view(-1).numpy())

    x = np.vstack(images_list)
    y = np.concatenate(labels_list)
    return x, y


def save_checkpoint(model, optimizer, epoch: int, path: str):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
    }
    torch.save(state, path)


def load_checkpoint(model, path: str, optimizer=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Чекпоінт не знайдено за шляхом: {path}")

    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('epoch', 0)


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_path: str):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
    ax1.plot(epochs, val_losses, label='Val Loss', color='orange')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, label='Train Accuracy', color='blue')
    ax2.plot(epochs, val_accs, label='Val Accuracy', color='orange')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def setup_logging(log_path: str = None, level=logging.INFO):
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_path:
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_environment_info() -> dict:
    import sklearn
    import torchvision
    info = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'sklearn': sklearn.__version__,
        'numpy': np.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
    return info


def save_history_csv(history: list, path: str):
    if not history:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
