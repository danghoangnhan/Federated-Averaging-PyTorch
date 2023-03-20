import json
import os

import flsim.configs  # noqa
import hydra
import torch
from flsim.data.data_sharder import SequentialSharder
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import fl_config_from_json
from flsim.utils.example_utils import (
    DataLoader,
    DataProvider,
    FLModel,
    MetricsReporter,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision import datasets, transforms

from configs.ILP_Heuristic_method_parameter import (
    num_of_original_client,
    num_of_head_client,
    num_of_MNIST_label,
)
from script.ResultToCSV import CreateResultData, Save_Accuracy_of_each_epoch
from script.getKL import saveKL
from src.algorithm.Heuristic import heuristic_method
from src.label import label_group
from src.model.CNN import MNIST_CNN
from src.sampling import mnist_noniid, mnist_heuristic
from src.visual import showLabelDistribution

IMAGE_SIZE = 28
total_execution_time = 0


def build_data_provider(local_batch_size,
                        examples_per_user,
                        drop_last: bool = False,
                        dataDir: str = "./data/Experiment/data/MNIST"):
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root=dataDir,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root=dataDir,
        train=False,
        download=True,
        transform=transform
    )
    client_num = num_of_original_client

    saveKL(train_dataset=train_dataset,
           label=num_of_MNIST_label,
           num_of_client=100,
           fileName="FL_non_IID_Heuristic_MNIST(CNN)"
           )

    dict_users = mnist_noniid(train_dataset, client_num)
    client_numnum_of_head_client = 20
    test = mnist_heuristic(dataset=train_dataset,
                           num_users=100,
                           headClient=client_numnum_of_head_client)
    print(test)
    origin = [train_dataset[i] for i in range(len(train_dataset))]
    label_per_group, labelcount = label_group(
        sorted_train_dataset=origin,
        groupSize=100
    )
    showLabelDistribution(labelcount,
                          fileName="before_heuristic")

    index_of_head_group, time = heuristic_method(
        train_dataset,
        num_of_MNIST_label,
        dict_users,
        client_num,
        num_of_head_client
    )

    global total_execution_time
    total_execution_time = total_execution_time + time

    sorted_train_dataset = []
    for group in index_of_head_group:
        for client_index in group:
            for data_index in dict_users[client_index]:
                sorted_train_dataset.append(train_dataset[int(data_index)])

    label_per_group, labelcount = label_group(
        sorted_train_dataset=sorted_train_dataset,
        groupSize=int(len(train_dataset) / examples_per_user)
    )
    showLabelDistribution(labelcount, "after_heuristic")
    saveKL(train_dataset=sorted_train_dataset,
           label=num_of_MNIST_label,
           num_of_client=int(len(train_dataset) / examples_per_user),
           fileName="FL_non_IID_Heuristic_MNIST(CNN)"
           )

    sharder = SequentialSharder(examples_per_shard=examples_per_user)

    fl_data_loader = DataLoader(
        sorted_train_dataset,
        test_dataset,
        test_dataset,
        sharder,
        local_batch_size,
        drop_last
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
    model = MNIST_CNN()
    # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
    global_model = FLModel(model, device)
    assert (global_model.fl_get_module() == model)

    if cuda_enabled:
        global_model.fl_cuda()

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

    Save_Accuracy_of_each_epoch(1,
                                "FL_non_IID_Heuristic_MNIST(CNN)",
                                accuracy_of_each_epoch,
                                best_accuracy_of_each_epoch)
    global total_execution_time
    client_num = num_of_original_client
    CreateResultData("FL_non_IID_Heuristic_MNIST(CNN)", "MNIST", "CNN", "non-IID -> IID", client_num,
                     int(trainer_config.epochs), eval_score['Accuracy'], total_execution_time)


@hydra.main(config_path="../newcode/configs", config_name="MNIST_config", version_base="1.2")
def run(cfg: DictConfig) -> None:
    print('-------------------FL_non_IID_Heuristic_MNIST(CNN)-------------------')
    trainer_config = cfg.trainer
    data_config = cfg.data
    main(
        trainer_config,
        data_config
    )


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    cfg = OmegaConf.create(fl_config_from_json(json.load(open(ROOT_DIR+'/configs/ILP_Heuristic_MNIST_config.json'))))
    run(cfg)
