# train.py
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import FocusLSTM

class AttentivenessTrainer:
    def __init__(self, data_path="data/processed/driver_attentiveness", model_path="models/driver_attentiveness.pth", batch_size=32, lr=0.001, epochs=12):
        self.data_path = data_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = FocusLSTM().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def load_data(self):
        X = np.load(os.path.join(self.data_path, "X_sequences.npy"))   # Shape: [N, 30, 4]
        y = np.load(os.path.join(self.data_path, "y_labels.npy"))      # Shape: [N]

        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        loader = self.load_data()
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                out = self.model(batch_x)
                loss = self.criterion(out, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch + 1}/{self.epochs}: Loss = {avg_loss:.4f}")
        self.save_model()

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved at {self.model_path}")

def main():
    trainer = AttentivenessTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
