import math
import time
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from config import Config
from utils import save_checkpoint, plot_learning_curves, save_history_csv

logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.view(-1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            targets = targets.view(-1).to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_pytorch_model(
    model,
    train_loader,
    val_loader,
    model_name: str,
    config: Config = Config
):
    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = None
    if config.SCHEDULER == 'cosine':
        steps_per_epoch = len(train_loader)
        total_steps = config.NUM_EPOCHS * steps_per_epoch
        warmup_steps = int(config.WARMUP_EPOCHS * steps_per_epoch)

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        logger.info(f"Scheduler: cosine, warmup {config.WARMUP_EPOCHS} епох, піковий lr {config.LEARNING_RATE}")

    best_val_acc = 0.0
    best_epoch = 0
    history = []
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    checkpoint_path = f"{config.CHECKPOINT_DIR}/{model_name}_best.pth"
    curve_path = f"{config.RESULTS_DIR}/{model_name}_learning_curves.png"
    history_path = f"{config.RESULTS_DIR}/{model_name}_history.csv"

    use_cuda = str(config.DEVICE).startswith('cuda') and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    logger.info(f"Початок навчання моделі {model_name} на {config.DEVICE}")
    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE, scheduler)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.DEVICE)
        epoch_time = time.time() - epoch_start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 6),
            'val_loss': round(val_loss, 6),
            'val_acc': round(val_acc, 6),
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time_s': round(epoch_time, 1),
            'is_best': is_best,
        })

        logger.info(
            f"Епоха [{epoch:02d}/{config.NUM_EPOCHS:02d}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"Час: {epoch_time:.1f} с"
            + (" | * найкраща (збережено чекпоінт)" if is_best else "")
        )

    total_time = time.time() - start_time
    peak_mem_mb = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1) if use_cuda else None

    logger.info(
        f"Навчання завершено за {total_time / 60:.1f} хв. "
        f"Найкраща епоха: {best_epoch} (Val Acc: {best_val_acc:.4f})"
        + (f". Пікова пам'ять GPU: {peak_mem_mb} МБ" if peak_mem_mb else "")
    )

    plot_learning_curves(train_losses, val_losses, train_accs, val_accs, curve_path)
    save_history_csv(history, history_path)
    logger.info(f"Криві навчання: {curve_path}")
    logger.info(f"Історія навчання по епохах: {history_path}")

    info = {
        'train_time_s': round(total_time, 1),
        'mean_epoch_time_s': round(total_time / config.NUM_EPOCHS, 1),
        'best_epoch': best_epoch,
        'best_val_acc': round(best_val_acc, 6),
        'peak_gpu_mem_mb': peak_mem_mb,
        'history_csv': history_path,
    }
    return model, info
