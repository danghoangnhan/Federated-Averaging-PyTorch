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
import flsim.configs  # noqa
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from flsim.data.data_sharder import SequentialSharder
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.config_utils import fl_config_from_json
from flsim.utils.example_utils import (
    DataLoader,
    DataProvider,
    FLModel,
    MetricsReporter,
    SimpleConvNet,
)
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from torchvision import datasets,transforms
from torch import Tensor


IMAGE_SIZE = 28

class MLP(nn.Module):
    def __init__(self):
       
        super(MLP, self).__init__()
        # number of hidden nodes in each layer (200)
        hidden = 200
        
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        #self.fc3 = nn.Linear(hidden_2, 10)
    
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        #x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

def build_data_provider(local_batch_size, examples_per_user, drop_last: bool = False):

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root="../Experiment/data/MNIST", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="../Experiment/data/MNIST", train=False, download=True, transform=transform
    )
    sharder = SequentialSharder(examples_per_shard=examples_per_user)
    fl_data_loader = DataLoader(
        train_dataset, test_dataset, test_dataset, sharder, local_batch_size, drop_last
    )
    data_provider = DataProvider(fl_data_loader)
    print(f"Clients in total: {data_provider.num_train_users()}")
    return data_provider


def main(
    trainer_config,
    data_config,
    use_cuda_if_available: bool = True,
) -> None:
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    model = MLP()
    # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
    global_model = FLModel(model, device)
    assert(global_model.fl_get_module() == model)

    if cuda_enabled:
        global_model.fl_cuda()
    #print(f"Created {trainer_config._target_}")
    data_provider = build_data_provider(
        local_batch_size=data_config.local_batch_size,
        examples_per_user=data_config.examples_per_user,
        drop_last=False,
    )
    
    #print(trainer_config)
    #print(data_config)
    
    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])
    
    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    
    #print(global_model)
    #print(model)
    #print(device)
    #print(data_provider)
    #print(metrics_reporter)
    #print(data_provider.num_train_users())
    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=1
    )

    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter([Channel.STDOUT]),
    )


@hydra.main(config_path="configs", config_name="MNIST_config" , version_base="1.2")
def run(cfg: DictConfig) -> None:
    
    print(cfg)
    trainer_config = cfg.trainer
    data_config = cfg.data
    main(
        trainer_config,
        data_config
    )


if __name__ == "__main__":
    
    f = open('C:\Python\Program\Experiment\configs\MNIST_config.json')
    data = json.load(f)
    json_cfg = fl_config_from_json(data)
    #print(cfg1)
    cfg = maybe_parse_json_config()
    cfg=OmegaConf.create(json_cfg)

    run(cfg)
