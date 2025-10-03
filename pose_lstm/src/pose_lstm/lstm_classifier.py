from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    num_classes: int = 2


class LSTMClassifier(nn.Module):
    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        lstm_out_size = config.hidden_size * (2 if config.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Linear(lstm_out_size, lstm_out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout),
            nn.Linear(lstm_out_size, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Use last time-step features
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits
