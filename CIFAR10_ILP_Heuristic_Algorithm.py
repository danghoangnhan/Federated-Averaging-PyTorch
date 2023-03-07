import os
import torch
import time
import numpy as np
import cvxpy as cp
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch import Tensor
from torch import arange
import matplotlib.pyplot as plt
import math
from torchvision.datasets.cifar import CIFAR10
from configs.little_case_ILP_Heuristic_method_parameter import (
    num_of_original_client,
    num_of_used_client,
    num_of_head_client,
    data_size_of_original_MNIST_client,
    data_size_of_original_CIFAR10_client,
    num_of_MNIST_label,
    num_of_CIFAR10_label,
    Max_value_of_ILP,
)
from script.getKL import get_KL_value
from src.algorithm.Heuristic import heuristic_method
from src.sampling import cifar_noniid
import itertools

from src.algorithm.ILP import ILP_method

IMAGE_SIZE = 28
heuristic_total_execution_time = 0
ILP_total_execution_time = 0


def KL_of_combination(num_of_label, total_dataset_size, P_i, P_j, Q):
    # P_i,P_j => list.

    KL = 0
    for i in range(num_of_label):
        P = (P_i[i] + P_j[i]) / total_dataset_size
        if P != 0:
            result_of_division = P / Q
            KL = KL + (P * math.log(result_of_division, 2))
    return KL


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = CIFAR10(
        root="../Experiment/data/cifar10", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR10(
        root="../Experiment/data/cifar10", train=False, download=True, transform=transform
    )
    # print(train_dataset)
    client_num = num_of_original_client
    # print(client_num)
    # divide train dataset(non-iid)

    dict_users = cifar_noniid(train_dataset, client_num)

    ##########heuristic method##########
    print("##########heuristic method##########")
    heuristic_index_of_head_group, time = heuristic_method(train_dataset,
                                                           num_of_CIFAR10_label, dict_users, client_num,
                                                           num_of_used_client, num_of_head_client)

    global heuristic_total_execution_time
    heuristic_total_execution_time = heuristic_total_execution_time + time

    # merge train dataset
    num_of_client = num_of_head_client
    sorted_heuristic_method_train_dataset = []
    # print(len(dict_users[0]))
    for k in range(num_of_head_client):

        for i in range(len(heuristic_index_of_head_group[k])):

            client_index = heuristic_index_of_head_group[k][i]
            # print(client_index)
            for j in range(len(dict_users[client_index])):
                data_index = int(dict_users[client_index][j])
                sorted_heuristic_method_train_dataset.append(train_dataset[data_index])
    heuristic_KL_of_each_client, heuristic_avg_KL = get_KL_value(sorted_heuristic_method_train_dataset,
                                                                 num_of_MNIST_label, num_of_client)
    # print(heuristic_avg_KL)
    ##########ILP method##########
    print("##########ILP method##########")
    ILP_index_of_head_group, ILP_time = ILP_method(train_dataset,
                                                   num_of_CIFAR10_label,
                                                   dict_users,
                                                   client_num,
                                                   num_of_used_client,
                                                   num_of_head_client,
                                                   Max_value_of_ILP
                                                   )

    global ILP_total_execution_time
    ILP_total_execution_time = ILP_total_execution_time + ILP_time

    # merge train dataset
    num_of_client = num_of_head_client
    sorted_ILP_method_train_dataset = []
    # print(len(dict_users[0]))
    for k in range(num_of_head_client):

        for i in range(len(ILP_index_of_head_group[k])):
            client_index = ILP_index_of_head_group[k][i]
            # print(client_index)
            for j in range(len(dict_users[client_index])):
                data_index = int(dict_users[client_index][j])
                sorted_ILP_method_train_dataset.append(train_dataset[data_index])
    ILP_KL_of_each_client, ILP_avg_KL = get_KL_value(sorted_ILP_method_train_dataset, num_of_CIFAR10_label,
                                                     num_of_client)
    # print(ILP_avg_KL)


if __name__ == "__main__":
    main()
