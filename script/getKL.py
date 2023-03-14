import math
import csv


def KL_function(num_of_label, dataset_size, P_i, Q):
    # P_i => list.

    KL = 0
    for i in range(num_of_label):
        P = P_i[i] / dataset_size
        if P != 0:
            result_of_division = P / Q
            KL = KL + (P * math.log(result_of_division, 2))
    return KL


def KL_of_combination(num_of_label,
                      total_dataset_size,
                      P_i,
                      P_j,
                      Q
                      ):
    KL = 0
    for i in range(num_of_label):
        P = (P_i[i] + P_j[i]) / total_dataset_size
        if P != 0:
            result_of_division = P / Q
            KL = KL + (P * math.log(result_of_division, 2))
    return KL


def get_KL_value(train_dataset,
                 num_of_label,
                 client_num
                 ):
    num_of_client = client_num

    nums_of_labels_of_each_client = []

    indexs_of_clients = []

    client_size = len(train_dataset) / num_of_client

    # KL setting
    KL_of_each_client = []

    label_num = num_of_label

    distribution_Q = 1 / label_num

    avg_KL = 0

    # Assigning the indexs of each client

    index = []
    for i in range(len(train_dataset)):
        index.append(i)

        if i > 0 and (i + 1) % (len(train_dataset) / num_of_client) == 0:
            indexs_of_clients.append(index)
            index = []

    # Counting the number of labels of each client
    for i in range(num_of_client):
        num_of_class_list = [0] * label_num

        for j in range(len(indexs_of_clients[i])):
            index_of_dataset = indexs_of_clients[i][j]
            label = train_dataset[index_of_dataset][1]
            num_of_class_list[label] = num_of_class_list[label] + 1

        nums_of_labels_of_each_client.append(num_of_class_list)

    # Calculating the KL of each client
    for i in range(num_of_client):
        KL = KL_function(label_num, client_size, nums_of_labels_of_each_client[i], distribution_Q)
        avg_KL = avg_KL + KL
        KL_of_each_client.append(KL)

    avg_KL = avg_KL / num_of_client
    # print(KL_of_each_client)
    print("AVG KL :", avg_KL)
    return KL_of_each_client, avg_KL


def Save_KL_Result(filename, KL_list, avg_KL):
    filepath = "./result/KLresult/" + filename + ".csv"

    header = ["Client_ID", "KL value", "", "Average KL"]
    data = []
    f = open(filepath, 'w+', encoding='UTF8', newline='')

    writer = csv.writer(f)

    writer.writerow(header)

    for i in range(len(KL_list)):
        data = [i, KL_list[i], ""]
        if i == 0:
            data.append(avg_KL)

        writer.writerow(data)
    f.close()


def saveKL(
        train_dataset,
        label,
        num_of_client,
        fileName: str = "FL_non_IID_Heuristic_cifar10(CNN)"):
    KL_of_each_client, avg_KL = get_KL_value(
        train_dataset,
        label,
        num_of_client
    )
    Save_KL_Result(fileName, KL_of_each_client, avg_KL)
