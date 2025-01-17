import json
import flsim.configs  # noqa
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from flsim.data.data_sharder import SequentialSharder
from flsim.data.data_sharder import RandomSharder
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
from torchvision import datasets, transforms
from torch import Tensor
from script.ResultToCSV import CreateHeader, CreateResultData, Save_KL_Result, Save_Accuracy_of_each_epoch
from script.getKL import get_KL_value
from model.MNIST_MLP import MNIST_MLP

IMAGE_SIZE = 28


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

    client_num = int(len(train_dataset) / examples_per_user)

    KL_of_each_client, avg_KL = get_KL_value(train_dataset, 10, client_num)

    Save_KL_Result("FL_IID_MNIST(MLP)", KL_of_each_client, avg_KL)
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
    model = MNIST_MLP()
    # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
    global_model = FLModel(model, device)
    assert (global_model.fl_get_module() == model)

    if cuda_enabled:
        global_model.fl_cuda()
    # print(f"Created {trainer_config._target_}")
    data_provider = build_data_provider(
        local_batch_size=data_config.local_batch_size,
        examples_per_user=data_config.examples_per_user,
        drop_last=False,
    )

    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)

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
    accuracy_of_each_epoch = metrics_reporter.AccuracyList
    best_accuracy_of_each_epoch = max(accuracy_of_each_epoch)
    # print("Accuracy list:",accuracy_of_each_epoch)
    # print("Best Accuracy:",best_accuracy_of_each_epoch)

    Save_Accuracy_of_each_epoch(1, "FL_IID_MNIST(MLP)", accuracy_of_each_epoch, best_accuracy_of_each_epoch)
    client_num = data_provider.num_train_users()
    CreateResultData("FL_IID_MNIST(MLP)", "MNIST", "MLP", "IID", client_num, int(trainer_config.epochs),
                     eval_score['Accuracy'], "")
    # print(eval_score)


@hydra.main(config_path="configs", config_name="MNIST_config", version_base="1.2")
def run(cfg: DictConfig) -> None:
    print('-------------------FL_IID_MNIST(MLP)-------------------')
    print(cfg)
    trainer_config = cfg.trainer
    data_config = cfg.data
    main(
        trainer_config,
        data_config
    )


if __name__ == "__main__":
    f = open('configs/MNIST_config.json')
    data = json.load(f)
    json_cfg = fl_config_from_json(data)
    # print(cfg1)
    cfg = maybe_parse_json_config()
    cfg = OmegaConf.create(json_cfg)
    run(cfg)