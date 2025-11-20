# Databricks notebook source
import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    """DeepConvNet for EEG classification (Schirrmeister et al., 2017)"""
    def __init__(self, n_classes=4, n_channels=22, n_samples=500):
        super(DeepConvNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples

        self.conv1 = nn.Conv2d(1, 25, (1, 5), padding=(0, 2))
        self.conv2 = nn.Conv2d(25, 25, (n_channels, 1))
        self.batchnorm1 = nn.BatchNorm2d(25)
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d((1, 2))
        self.dropout1 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(25, 50, (1, 5), padding=(0, 2))
        self.batchnorm2 = nn.BatchNorm2d(50)
        self.elu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d((1, 2))
        self.dropout2 = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(50, 100, (1, 5), padding=(0, 2))
        self.batchnorm3 = nn.BatchNorm2d(100)
        self.elu3 = nn.ELU()
        self.pool3 = nn.MaxPool2d((1, 2))
        self.dropout3 = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(100, 200, (1, 5), padding=(0, 2))
        self.batchnorm4 = nn.BatchNorm2d(200)
        self.elu4 = nn.ELU()
        self.pool4 = nn.MaxPool2d((1, 2))
        self.dropout4 = nn.Dropout(0.5)

        # Compute output size after convolutions and pooling
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
            x = self.conv3(x)
            x = self.batchnorm2(x)
            x = self.elu2(x)
            x = self.pool2(x)
            x = self.dropout2(x)
            x = self.conv4(x)
            x = self.batchnorm3(x)
            x = self.elu3(x)
            x = self.pool3(x)
            x = self.dropout3(x)
            x = self.conv5(x)
            x = self.batchnorm4(x)
            x = self.elu4(x)
            x = self.pool4(x)
            x = self.dropout4(x)
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
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.elu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.conv5(x)
        x = self.batchnorm4(x)
        x = self.elu4(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x