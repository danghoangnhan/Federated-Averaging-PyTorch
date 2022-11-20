#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In this tutorial, we will train an image classifier with FLSim to simulate a federated learning training environment.

With this tutorial, you will learn the following key components of FLSim:
1. Data loading
2. Model construction
3. Trainer construction

    Typical usage example:
    python3 cifar10_example.py --config-file configs/cifar10_config.json
"""
import json
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2)
        
        self.gn_relu = nn.Sequential(
            nn.GroupNorm(32, 32, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout =nn.Dropout(p=0, inplace=False)
        self.fc1 = nn.Linear(288, 10, bias=True)

    def forward(self, x):
        x = self.gn_relu(self.conv1(x))
        x = self.gn_relu(self.conv2(x))
        x = self.gn_relu(self.conv3(x))
        x = self.gn_relu(self.conv4(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = self.fc1(x)
        return x
