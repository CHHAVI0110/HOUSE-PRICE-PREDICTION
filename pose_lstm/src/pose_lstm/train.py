from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
from rich.progress import Progress
from rich.console import Console

from pose_lstm.dataset import PoseSequenceDataset, SequenceConfig
from pose_lstm.lstm_classifier import LSTMClassifier, LSTMConfig


console = Console()


@dataclass
class TrainConfig:
    data_root: str = "data_npy"
    sequence_length: int = 30
    step: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    max_epochs: int = 20
    num_workers: int = 2
    val_split: float = 0.2
    seed: int = 42
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    num_classes: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_input_size(data_root: str) -> int:
    # Find first .npy to infer feature dim
    for dirpath, _, filenames in os.walk(data_root):
        for fname in filenames:
            if fname.endswith(".npy"):
                arr = np.load(os.path.join(dirpath, fname))
                return int(arr.shape[1])
    raise RuntimeError("No .npy files found to infer input size")


def create_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, int]:
    seq_cfg = SequenceConfig(sequence_length=cfg.sequence_length, step=cfg.step)
    dataset = PoseSequenceDataset(cfg.data_root, seq_cfg)

    num_classes = len(dataset.class_to_idx)

    val_len = int(len(dataset) * cfg.val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return train_loader, val_loader, num_classes


def train() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    train_loader, val_loader, num_classes = create_dataloaders(cfg)
    input_size = infer_input_size(cfg.data_root)

    model_cfg = LSTMConfig(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
        num_classes=num_classes,
    )
    model = LSTMClassifier(model_cfg).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    best_val_acc = 0.0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        with Progress() as progress:
            task = progress.add_task(f"[green]Epoch {epoch}/{cfg.max_epochs}", total=len(train_loader))
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(cfg.device)
                batch_y = batch_y.to(cfg.device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * batch_x.size(0)
                preds = logits.argmax(dim=1)
                correct += int((preds == batch_y).sum().item())
                total += int(batch_x.size(0))
                progress.update(task, advance=1)

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        v_total, v_correct, v_loss_total = 0, 0, 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(cfg.device)
                batch_y = batch_y.to(cfg.device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                v_loss_total += float(loss.item()) * batch_x.size(0)
                preds = logits.argmax(dim=1)
                v_correct += int((preds == batch_y).sum().item())
                v_total += int(batch_x.size(0))
                y_true.extend(batch_y.tolist())
                y_pred.extend(preds.tolist())

        val_loss = v_loss_total / max(1, v_total)
        val_acc = v_correct / max(1, v_total)

        console.print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "config": model_cfg.__dict__,
                "input_size": input_size,
                "sequence_length": cfg.sequence_length,
                "class_to_idx": getattr(train_loader.dataset.dataset, "class_to_idx", {}),
            }, os.path.join("models", "best_lstm.pth"))
            console.print(f"Saved new best model with val_acc={best_val_acc:.4f}")

        scheduler.step()

    # Final report on validation set
    console.print("\nValidation classification report:")
    console.print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    train()
