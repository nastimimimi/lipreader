"""
lipreader/src/model.py
Conv3D frontend + BiLSTM classifier (уменьшенная версия для малого датасета)
"""

import sys, os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SEQUENCE_LENGTH, FRAME_H, FRAME_W,
    NUM_CLASSES, LSTM_HIDDEN, LSTM_LAYERS,
    LSTM_DROPOUT, FC_DROPOUT
)


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3,3), pool=(1,2,2)):
        super().__init__()
        pad = tuple(k // 2 for k in kernel)
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool3d(pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LipReader(nn.Module):
    """
    Вход  : (B, 1, T, H, W)
    Выход : (B, num_classes)
    """

    def __init__(self,
                 num_classes=NUM_CLASSES,
                 lstm_hidden=LSTM_HIDDEN,
                 lstm_layers=LSTM_LAYERS,
                 lstm_dropout=LSTM_DROPOUT,
                 fc_dropout=FC_DROPOUT):
        super().__init__()

        # Conv3D frontend — маленький для малого датасета
        self.frontend = nn.Sequential(
            ConvBlock3D(1,  16, pool=(1, 2, 2)),   # (B, 16, T, 56, 56)
            ConvBlock3D(16, 32, pool=(1, 2, 2)),   # (B, 32, T, 28, 28)
            ConvBlock3D(32, 64, pool=(1, 2, 2)),   # (B, 64, T, 14, 14)
        )

        # Пространственное усреднение 14×14 → 1×1
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # BiLSTM — input_size=64 совпадает с последним ConvBlock
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(lstm_hidden * 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.frontend(x)                  # (B, 64, T, 14, 14)
        x = self.spatial_pool(x)              # (B, 64, T, 1, 1)
        x = x.squeeze(-1).squeeze(-1)         # (B, 64, T)
        x = x.permute(0, 2, 1)               # (B, T, 64)
        x, _ = self.lstm(x)                   # (B, T, hidden*2)
        x = x[:, -1, :]                       # (B, hidden*2)
        return self.classifier(x)             # (B, num_classes)

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LipReader()
    print(f"Параметров: {count_parameters(model):,}")
    dummy = torch.zeros(2, 1, SEQUENCE_LENGTH, FRAME_H, FRAME_W)
    out   = model(dummy)
    print(f"Вход: {dummy.shape}  →  Выход: {out.shape}")