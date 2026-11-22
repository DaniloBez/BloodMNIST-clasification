import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

from config import Config
from models.base_model import BaseModel
from train import train_pytorch_model
from evaluate import evaluate_pytorch_model
from utils import load_checkpoint


class CNNModel(BaseModel):
    def __init__(self, name: str = "CNN_28", config: Config = Config):
        super().__init__(name, config)
        self.net = resnet18(weights=ResNet18_Weights.DEFAULT)

        if config.IMAGE_SIZE == 28:
            self.net.conv1 = nn.Conv2d(
                in_channels=config.IN_CHANNELS,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
            self.net.maxpool = nn.Identity()

        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, config.NUM_CLASSES)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
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
        return evaluate_pytorch_model(self.net, data_loader, device=self.config.DEVICE)

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.config.DEVICE))