import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split

class EmotionTrainer:
    def __init__(self, dataset_path="src/data/raw/emotion", model_path="models/emotion_model.pth", batch_size=64, lr=1e-4, epochs=20, patience=5):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(pretrained=True)
        self.num_classes = self._get_num_classes()
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    def _get_num_classes(self):
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = ImageFolder(self.dataset_path, transform=transform)
        return len(dataset.classes)

    def _load_dataloaders(self):
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = ImageFolder(self.dataset_path, transform=transform)

        val_split = 0.2
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        return train_loader, val_loader

    def train(self):
        train_loader, val_loader = self._load_dataloaders()
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_y.size(0)

            train_acc = total_correct / total_samples
            val_acc, val_loss = self.validate(val_loader)

            print(f"Epoch {epoch + 1}/{self.epochs} => Train Loss: {total_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            self.scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()
                print(f" Best model saved with val acc: {best_val_acc:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(" Early stopping triggered.")
                    break

        print(" Training complete.")

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total, val_loss / len(val_loader)

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f" Model saved at {self.model_path}")

def main():
    trainer = EmotionTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
