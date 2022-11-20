
from config.ILP_Heuristic_method_parameter import (
    num_of_original_client,
    num_of_head_client,
    data_size_of_original_MNIST_client,
    data_size_of_original_CIFAR10_client,
    num_of_MNIST_label,
    num_of_CIFAR10_label,
    Max_value_of_ILP,           
)
from algorithm.Heuristic_Algorithm import heuristic_method
from algorithm.ILP_Algorithm import ILP_method

ILP_Heuristic_

, Save_Accuracy_of_each_epoch

    accuracy_of_each_epoch = metrics_reporter.AccuracyList
    best_accuracy_of_each_epoch = max(accuracy_of_each_epoch)
    #print("Accuracy list:",accuracy_of_each_epoch)
    #print("Best Accuracy:",best_accuracy_of_each_epoch)

    Save_Accuracy_of_each_epoch(mode, casename, accuracy_of_each_epoch,best_accuracy_of_each_epoch)