import os
import logging
from tqdm import tqdm
from typing import Tuple
import torch
from torch.utils.data import DataLoader

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class FacialExpressionFeatureBuilder:
    """
    Feature pipeline for Facial Micro Expression.
    Prepares dataloaders from train and val datasets.
    """

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def create_dataloaders(self, train_dataset, val_dataset) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch dataloaders for training and validation datasets.

        Args:
            train_dataset: torch.utils.data.Dataset
            val_dataset: torch.utils.data.Dataset

        Returns:
            Tuple containing (train_loader, val_loader)
        """
        logger.info("Creating dataloaders...")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        logger.info(f"Dataloaders created | Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        return train_loader, val_loader


# ========================
# Main Execution
# ========================
def main():
    from src.preprocessing.emotion_preprocessor import FacialEmotionPreprocessor

    logger.info("Loading dataset via FacialEmotionPreprocessor...")
    preprocessor = FacialEmotionPreprocessor()
    train_data, val_data, _ = preprocessor.load_dataset()

    feature_builder = FacialExpressionFeatureBuilder(batch_size=64)
    train_loader, val_loader = feature_builder.create_dataloaders(train_data, val_data)

    logger.info("Feature pipeline setup complete.")


if __name__ == "__main__":
    main()
