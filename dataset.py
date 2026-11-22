import torchvision.transforms as transform
from torch.utils.data import DataLoader
from medmnist import BloodMNIST
from config import Config


def get_transforms(config: Config = Config):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    augmentations = []
    if config.AUGMENTATION == 'crop':
        # м'який випадковий кроп із масштабуванням назад до вихідного розміру
        augmentations.append(transform.RandomResizedCrop(
            size=config.IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)))
    augmentations += [
        transform.RandomHorizontalFlip(p=0.5),
        transform.RandomVerticalFlip(p=0.5),
        transform.RandomRotation(degrees=15),
    ]
    if config.AUGMENTATION == 'color':
        # слабкий colour jitter, без зсуву відтінку (колір клітин діагностично важливий)
        augmentations.append(transform.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))

    train_transform = transform.Compose(augmentations + [
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    val_test_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_test_transform


def get_dataloaders(config: Config = Config, augment: bool = True):
    train_tf, val_test_tf = get_transforms(config)
    if not augment:
        train_tf = val_test_tf

    train_dataset = BloodMNIST(
        split='train',
        transform=train_tf,
        download=True,
        as_rgb=True,
        root=config.DATA_DIR,
        size=config.IMAGE_SIZE  # 28 або 224
    )

    val_dataset = BloodMNIST(
        split='val',
        transform=val_test_tf,
        download=True,
        as_rgb=True,
        root=config.DATA_DIR,
        size=config.IMAGE_SIZE
    )

    test_dataset = BloodMNIST(
        split='test',
        transform=val_test_tf,
        download=True,
        as_rgb=True,
        root=config.DATA_DIR,
        size=config.IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == 'cuda'),
        persistent_workers=(config.NUM_WORKERS > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(config.DEVICE == 'cuda')
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(config.DEVICE == 'cuda')
    )

    return train_loader, val_loader, test_loader