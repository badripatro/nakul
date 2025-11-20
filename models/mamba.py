# Databricks notebook source
import torch
import torch.nn as nn
from mamba_ssm import Mamba
class MambaEEGNet(nn.Module):
    """Mamba (sequence mixing) for EEG classification (adapted for BCI-IV-2a)"""
    def __init__(self, n_classes=4, n_channels=22, n_samples=500, dropout_rate=0.5):
        super(MambaEEGNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        # Initial temporal convolution
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_rate)

        # Sequence mixing block (Mamba-style) using mamba_ssm library
        self.mamba = Mamba(d_model=64)

        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (batch, channels, samples)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        # Mamba expects (batch, seq_len, d_model), so transpose
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        x = self.mamba(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x