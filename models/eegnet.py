# Databricks notebook source
import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """EEGNet for EEG classification (Lawhern et al., 2018)"""
    def __init__(self, n_classes=4, n_channels=22, n_samples=500, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        # First temporal convolution
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)

        # Depthwise convolution (spatial filtering)
        self.depthwiseConv = nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(self.dropout_rate)

        # Separable convolution
        self.separableConv = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(self.dropout_rate)

        # Fully connected layer
        self._output_dim = self._get_output_dim()
        self.fc = nn.Linear(self._output_dim, n_classes)

    def _get_output_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.depthwiseConv(x)
            x = self.batchnorm2(x)
            x = self.elu1(x)
            x = self.pool1(x)
            x = self.dropout1(x)
            x = self.separableConv(x)
            x = self.batchnorm3(x)
            x = self.elu2(x)
            x = self.pool2(x)
            x = self.dropout2(x)
            return x.numel()
    def forward(self, x):
        # x: (batch, channels, samples)
        x = x.unsqueeze(1)  # (batch, 1, channels, samples)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x