#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from torchvision import datasets, transforms
from torch import Tensor


class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        # number of hidden nodes in each layer (200)
        hidden = 200

        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        # self.fc3 = nn.Linear(hidden_2, 10)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
