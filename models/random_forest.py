import os
import time
import logging
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

from config import Config
from utils import prepare_rf_data
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    def __init__(self, name: str = "RF_28", config: Config = Config):
        super().__init__(name, config)

        self.model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            n_jobs=config.RF_N_JOBS,
            random_state=config.SEED,
            verbose=1
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        logger.info(f"Навчання Random Forest ({self.name})")
        logger.info("Конвертація зображень навчальної вибірки у 1D-вектори...")
        x_train, y_train = prepare_rf_data(train_loader)

        logger.info(f"Розмірність навчальної вибірки: {x_train.shape}")
        logger.info(f"Об'єктів: {x_train.shape[0]}, Ознак (пікселів): {x_train.shape[1]}")
        logger.info("Побудова ансамблю дерев рішень...")

        start_time = time.time()
        self.model.fit(x_train, y_train)
        train_time = time.time() - start_time
        n_nodes = int(sum(est.tree_.node_count for est in self.model.estimators_))
        self.train_info = {
            'train_time_s': round(train_time, 1),
            'n_trees': len(self.model.estimators_),
            'n_tree_nodes': n_nodes,
        }
        logger.info(f"Навчання Random Forest завершено за {train_time:.1f} с (вузлів у деревах: {n_nodes})")

        save_path = os.path.join(self.config.CHECKPOINT_DIR, f"{self.name}_best.joblib")
        self.save(save_path)

    def predict(self, data_loader: DataLoader):
        x, y_true = prepare_rf_data(data_loader)
        y_probs = self.model.predict_proba(x)
        y_pred = np.argmax(y_probs, axis=1)

        return y_true, y_pred, y_probs

    def save(self, path: str):
        joblib.dump(self.model, path)
        logger.info(f"Модель Random Forest збережено у: {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл моделі не знайдено за шляхом: {path}")

        self.model = joblib.load(path)
        logger.info(f"Модель Random Forest завантажено з: {path}")