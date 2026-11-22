import os
import torch

class Config:
    DATASET_NAME = 'bloodmnist'
    DATA_DIR = './data'
    IMAGE_SIZE = 28         # 28 або 224
    IN_CHANNELS = 3
    NUM_CLASSES = 8

    SEED = 323
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4

    BATCH_SIZE = 128 if IMAGE_SIZE == 28 else 32
    NUM_EPOCHS = 12
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 1e-4
    SCHEDULER = 'none'       # 'none' (постійний lr) або 'cosine' (warmup + косинусне спадання)
    WARMUP_EPOCHS = 1        # використовується лише при SCHEDULER='cosine'
    AUGMENTATION = 'basic'   # 'basic' (flip+rotation), 'crop' (+RandomResizedCrop), 'color' (+ColorJitter)

    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = None
    RF_N_JOBS = -1

    CHECKPOINT_DIR = './checkpoints'
    RESULTS_DIR = './results'

    # значення за замовчуванням для параметрів, які можна перевизначати в експериментах
    _DEFAULTS = dict(SEED=SEED, NUM_EPOCHS=NUM_EPOCHS, LEARNING_RATE=LEARNING_RATE,
                     SCHEDULER=SCHEDULER, AUGMENTATION=AUGMENTATION)

    @classmethod
    def configure(cls, seed=None, epochs=None, lr=None, scheduler=None, augmentation=None):
        """Скидає перевизначувані параметри до значень за замовчуванням і застосовує нові.
        Потрібно, бо Config є спільним для всіх запусків в одному процесі."""
        for key, value in cls._DEFAULTS.items():
            setattr(cls, key, value)
        if seed is not None:
            cls.SEED = seed
        if epochs is not None:
            cls.NUM_EPOCHS = epochs
        if lr is not None:
            cls.LEARNING_RATE = lr
        if scheduler is not None:
            cls.SCHEDULER = scheduler
        if augmentation is not None:
            cls.AUGMENTATION = augmentation

    @classmethod
    def setup(cls, image_size: int = None):
        if image_size in (28, 224):
            cls.IMAGE_SIZE = image_size
            cls.BATCH_SIZE = 128 if image_size == 28 else 32

        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)