import os
import numpy as np
from tqdm import tqdm
import warnings
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from PIL import Image

warnings.filterwarnings('ignore')

# ========================
# Configuration Constants
# ========================
RAW_DATA_PATH = "src/data/raw/facial_micro_expression"
PROCESSED_DATA_PATH = "src/data/processed/facial_micro_expression"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


# ========================
# Preprocessor Class
# ========================
class FacialEmotionPreprocessor:
    def __init__(self, val_split=0.2):
        self.val_split = val_split
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def load_dataset(self):
        dataset = ImageFolder(RAW_DATA_PATH, transform=self.transform)
        self.class_names = dataset.classes
        num_classes = len(self.class_names)

        train_size = int((1 - self.val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        return train_dataset, val_dataset, num_classes

    def save_class_mapping(self):
        mapping_path = os.path.join(PROCESSED_DATA_PATH, "class_mapping.txt")
        with open(mapping_path, "w") as f:
            for idx, class_name in enumerate(self.class_names):
                f.write(f"{idx}: {class_name}\n")
        print(f"Saved class mapping to {mapping_path}")


# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    preprocessor = FacialEmotionPreprocessor()
    train_data, val_data, num_classes = preprocessor.load_dataset()
    preprocessor.save_class_mapping()
    print(f"Loaded data with {num_classes} classes.")
    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
