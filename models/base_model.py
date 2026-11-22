from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from torch.utils.data import DataLoader
from config import Config


class BaseModel(ABC):
    def __init__(self, name: str, config: Config = Config):
        self.name = name
        self.config = config
        self.train_info = {}

    @abstractmethod
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        pass

    @abstractmethod
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass