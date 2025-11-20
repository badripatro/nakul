# Databricks notebook source
import torch
import torch.nn as nn

class ShallowConvNet(nn.Module):
    """ShallowConvNet for EEG classification (Schirrmeister et al., 2017)"""
    def __init__(self, n_classes=4, n_channels=22, n_samples=500, dropout_rate=0.5):
        super(ShallowConvNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(1, 40, (1, 13), padding=(0, 6))
        self.conv2 = nn.Conv2d(40, 40, (n_channels, 1))
        self.batchnorm1 = nn.BatchNorm2d(40)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.dropout1 = nn.Dropout(self.dropout_rate)

        self._output_dim = self._get_output_dim()
        self.fc = nn.Linear(self._output_dim, n_classes)

    def _get_output_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.batchnorm1(x)
            x = self.elu1(x)
            x = self.pool1(x)
            x = self.dropout1(x)
            return x.numel()
    def forward(self, x):
        # x: (batch, channels, samples)
        x = x.unsqueeze(1)  # (batch, 1, channels, samples)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x