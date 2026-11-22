import os
import logging
import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights, vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader

from config import Config
from models.base_model import BaseModel
from train import train_pytorch_model
from evaluate import evaluate_pytorch_model
from utils import load_checkpoint

logger = logging.getLogger(__name__)

TRANSFORMER_VARIANTS = ('vit', 'swin')


class TransformerModel(BaseModel):
    def __init__(self, name: str = "SWIN_224", variant: str = "swin", config: Config = Config):
        super().__init__(name, config)
        if variant not in TRANSFORMER_VARIANTS:
            raise ValueError(f"Невідомий варіант трансформера: {variant}. Доступні: {TRANSFORMER_VARIANTS}")
        self.variant = variant

        if variant == 'vit':
            self.net = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            in_features = self.net.heads.head.in_features
            self.net.heads.head = nn.Linear(in_features, config.NUM_CLASSES)
        else:
            self.net = swin_t(weights=Swin_T_Weights.DEFAULT)
            in_features = self.net.head.in_features
            self.net.head = nn.Linear(in_features, config.NUM_CLASSES)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        title = "ViT-B/16" if self.variant == 'vit' else "Swin-T"
        logger.info(f"Донавчання {title} ({self.name})")
        self.net, self.train_info = train_pytorch_model(
            model=self.net,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=self.name,
            config=self.config
        )

    def predict(self, data_loader: DataLoader):
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, f"{self.name}_best.pth")
        load_checkpoint(self.net, checkpoint_path)

        return evaluate_pytorch_model(
            self.net,
            data_loader,
            device=self.config.DEVICE
        )

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.config.DEVICE))
