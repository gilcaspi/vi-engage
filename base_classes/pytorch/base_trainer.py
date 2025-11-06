import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any

from base_classes.pytorch.early_stopping import EarlyStopping, EarlyStoppingMode


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 20,
        lr: float = 1e-3,
        early_stopping_patience: int = 5,
        min_delta: float = 0.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        self.early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=min_delta,
            mode=EarlyStoppingMode.MIN,
        )

    def train(self) -> Dict[str, Any]:
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in tqdm(range(self.epochs), desc="Training"):
            self.model.train()
            epoch_loss = 0.0

            for x_batch, y_batch in self.train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(self.train_loader)
            val_loss, val_acc = self.evaluate()
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            tqdm.write(
                f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Accuracy={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()

            self.early_stopper(val_loss)
            if self.early_stopper.early_stop:
                tqdm.write(f"Early stopping triggered at epoch {epoch + 1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.history

    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        correct, total = 0, 0
        total_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                total_loss += loss.item()
                predictions = output.argmax(dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy
