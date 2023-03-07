import math
import time
from script.getKL import get_KL_value

def heuristic_method(train_dataset,
                     num_of_label,
                     client_data_index,
                     original_client_num,
                     head_num):
    total_time = 0

    # client information
    num_of_client = original_client_num

    indexs_of_clients = []

    nums_of_labels_of_each_client = []

    client_size = len(train_dataset) / num_of_client

    # KL setting for splitting member and head
    KL_of_each_client = []

    label_num = num_of_label

    distribution_Q = 1 / label_num

    num_of_head = head_num

    num_of_member = num_of_client - num_of_head

    head_size = client_size

    member_size = client_size

    # Assigning the indexs of each client

    for i in range(num_of_client):
        index_lsit = []
        for j in range(len(client_data_index[i])):
            index_lsit.append(int(client_data_index[i][j]))

        indexs_of_clients.append(index_lsit)

    # Counting the number of labels of each client
    for i in range(num_of_client):
        num_of_class_list = [0] * label_num

        for j in range(len(indexs_of_clients[i])):
            index_of_original_dataset = int(indexs_of_clients[i][j])
            label = train_dataset[index_of_original_dataset][1]
            num_of_class_list[label] = num_of_class_list[label] + 1

        nums_of_labels_of_each_client.append(num_of_class_list)

    # print(len(nums_of_labels_of_each_client))
    # print(nums_of_labels_of_each_client[0])

    # Calculating the KL of each client
    for i in range(num_of_client):
        KL = 0
        for j in range(len(nums_of_labels_of_each_client[i])):
            distribution_P = nums_of_labels_of_each_client[i][j] / (len(train_dataset) / num_of_client)

            if distribution_P != 0:
                result_of_division = distribution_P / distribution_Q
                KL = KL + (distribution_P * math.log(result_of_division, 2))
        KL_of_each_client.append(KL)

    # print(len(KL_of_each_client))
    # print(KL_of_each_client)

    ####splitting the head and member####
    #
    head_index = []
    member_index = []
    for i in range(num_of_client):
        member_index.append(i)

    sorted_KL_list = sorted(KL_of_each_client)
    # print(sorted_KL_list)

    # find the heads

    for i in range(num_of_head):

        for j in range(len(member_index)):
            index_of_member = member_index[j]
            if (KL_of_each_client[index_of_member] == sorted_KL_list[i]) and (index_of_member not in head_index):
                index_of_head = index_of_member
                # print(index_of_head)
                head_index.append(index_of_head)
                member_index.remove(index_of_head)
                break

    ####heuristic algorithm
    final_size = int(len(train_dataset) / num_of_head)
    total_round = int((final_size - head_size) / member_size)
    head_index_group = []
    unassigned_member_index = member_index
    # convert to 2d list
    for i in range(num_of_head):
        index_list = []
        index_list.append(head_index[i])
        head_index_group.append(index_list)

    start_time = time.process_time()
    # assign the member with the lowest KL to a head
    for k in range(total_round):
        start_time = time.process_time()
        for i in range(num_of_head):

            KL_list = []
            size_of_combination_head = 1 * head_size + (len(head_index_group[i]) - 1) * member_size

            P_i_list = [0] * label_num

            for j in range(len(head_index_group[i])):
                for t in range(label_num):
                    P_i_list[t] = P_i_list[t] + nums_of_labels_of_each_client[head_index_group[i][j]][t]
            for j in range(len(unassigned_member_index)):
                P_j_list = nums_of_labels_of_each_client[unassigned_member_index[j]]
                KL = get_KL_value(label_num,
                                       size_of_combination_head + member_size,
                                       P_i_list,
                                       P_j_list,
                                       distribution_Q
                                       )
                KL_list.append(KL)

            # find the member with the lowest KL
            min_KL = min(KL_list)
            index_of_member_with_min_KL = KL_list.index(min_KL)
            # if k == 3 and (i<3):
            # print(KL_list)
            # print(min_KL)
            # print(index_of_member_with_min_KL)
            # assign the member to the head group
            head_index_group[i].append(unassigned_member_index[index_of_member_with_min_KL])
            unassigned_member_index.remove(unassigned_member_index[index_of_member_with_min_KL])
        end_time = time.process_time()
        total_time = total_time + (end_time - start_time)

    for i in range(num_of_head):
        print(head_index_group[i])

    print(total_time, " second")

    return head_index_group, total_time
