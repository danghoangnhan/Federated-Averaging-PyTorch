import itertools
import math
import time

import cvxpy as cp
import numpy as np
import torch
from src.sampling import mnist_noniid
from torchvision import datasets, transforms

from configs.little_case_ILP_Heuristic_method_parameter import (
    num_of_original_client,
    num_of_used_client,
    num_of_head_client,
    num_of_MNIST_label,
    Max_value_of_ILP,
)
from script.getKL import get_KL_value

IMAGE_SIZE=28
heuristic_total_execution_time = 0
ILP_total_execution_time = 0

def KL_of_combination(num_of_label,total_dataset_size,P_i,P_j,Q):
    #P_i,P_j => list.
    
    KL = 0
    for i in range(num_of_label):
        P = (P_i[i]+P_j[i])/total_dataset_size
        if P!=0:
            result_of_division = P / Q
            KL = KL + (P * math.log( result_of_division ,2))
    return KL



def heuristic_method(train_dataset, num_of_label, client_data_index, original_client_num, used_client_num, head_num):

    total_time = 0
    
    #client information
    num_of_client = used_client_num

    indexs_of_clients=[]

    nums_of_labels_of_each_client= []

    client_size=len(train_dataset)/original_client_num

    #KL setting for splitting member and head
    KL_of_each_client = []

    label_num = num_of_label

    distribution_Q = 1/label_num

    num_of_head = head_num

    num_of_member = num_of_client-num_of_head

    head_size=client_size

    member_size=client_size

	
    #Assigning the indexs of each client
    
    for i in range(num_of_client):
        index_lsit=[]
        for j in range(len(client_data_index[i])):
            index_lsit.append(int(client_data_index[i][j]))
    
        indexs_of_clients.append(index_lsit)
            
               
    #print(len(indexs_of_clients))
    #print(len(indexs_of_clients[0]))
    #print(indexs_of_clients[0])
    #print(len(indexs_of_clients[99]))

    #Counting the number of labels of each client
    for i in range(num_of_client):
        num_of_class_list = [0] * label_num
        
        for j in range(len(indexs_of_clients[i])):
            index_of_original_dataset = int(indexs_of_clients[i][j])
            label = train_dataset[index_of_original_dataset][1]
            num_of_class_list[label]=num_of_class_list[label]+1
            
        nums_of_labels_of_each_client.append(num_of_class_list)
        
    #print(len(nums_of_labels_of_each_client))
    #print(nums_of_labels_of_each_client[0])

    #Calculating the KL of each client
    for i in range(num_of_client):
        KL = 0
        for j in range(len(nums_of_labels_of_each_client[i])):
            distribution_P = nums_of_labels_of_each_client[i][j]/(len(train_dataset)/num_of_client)
            
            if distribution_P!=0:
                result_of_division = distribution_P / distribution_Q
                KL = KL + (distribution_P * math.log( result_of_division ,2))
        KL_of_each_client.append(KL)

    #print(len(KL_of_each_client))
    #print(KL_of_each_client)

    ####splitting the head and member####
    #
    head_index = []
    member_index = []
    for i in range(num_of_client):
        member_index.append(i)

    sorted_KL_list = sorted(KL_of_each_client)
    #print(sorted_KL_list)

    #find the heads
    
    for i in range(num_of_head):
        
        for j in range(len(member_index)):
            index_of_member = member_index[j]
            if (KL_of_each_client[index_of_member]==sorted_KL_list[i]) and (index_of_member not in head_index):
                index_of_head = index_of_member
                #print(index_of_head)
                head_index.append(index_of_head)
                member_index.remove(index_of_head)
                break

    #print(len(head_index))
    #print(len(member_index))
    #print(head_index)
    #print(member_index)

    #for i in range(len(head_index)):
        #print(KL_of_each_client[head_index[i]])
    #print("member")
    #for i in range(len(member_index)):
        #print(KL_of_each_client[member_index[i]])


    ####heuristic algorithm
    final_size = int((used_client_num*client_size)/num_of_head)
    total_round = int((final_size-head_size)/member_size)
    head_index_group = []
    unassigned_member_index = member_index
    #convert to 2d list
    for i in range(num_of_head):
        index_list=[]
        index_list.append(head_index[i])
        head_index_group.append(index_list)

        
    start_time = time.process_time()
    #assign the member with the lowest KL to a head
    for k in range(total_round):
        start_time = time.process_time()
        for i in range(num_of_head):
            
            KL_list=[]
            size_of_combination_head = 1 * head_size + (len(head_index_group[i])-1) * member_size
            
            P_i_list=[0] * label_num
            
            for j in range(len(head_index_group[i])):
                for t in range(label_num):
                    P_i_list[t] = P_i_list[t] + nums_of_labels_of_each_client[head_index_group[i][j]][t]
            for j in range(len(unassigned_member_index)):
                P_j_list = nums_of_labels_of_each_client[unassigned_member_index[j]]
                KL = KL_of_combination(label_num,size_of_combination_head+member_size,P_i_list,P_j_list,distribution_Q)
                KL_list.append(KL)
               
            #find the member with the lowest KL
            min_KL = min(KL_list)
            index_of_member_with_min_KL = KL_list.index(min_KL)
            #if k == 3 and (i<3):
                #print(KL_list)
                #print(min_KL)
                #print(index_of_member_with_min_KL)
            #assign the member to the head group
            head_index_group[i].append(unassigned_member_index[index_of_member_with_min_KL])
            unassigned_member_index.remove(unassigned_member_index[index_of_member_with_min_KL])
        end_time = time.process_time()
        total_time = total_time + (end_time-start_time)

    for i in range(num_of_head):
        print(head_index_group[i])
        
    print(total_time," seconds")

    
    return head_index_group, total_time

def ILP_method(train_dataset, num_of_label, client_data_index, original_client_num, used_client_num, head_num, ILP_Max_value):

    total_time = 0
    #client information
    num_of_client = used_client_num

    indexs_of_clients=[]

    nums_of_labels_of_each_client= []

    client_size=len(train_dataset)/original_client_num

    #KL setting for splitting member and head

    selected_member_num = 2
    
    KL_of_each_client = []

    label_num = num_of_label

    distribution_Q = 1/label_num

    num_of_head = head_num

    num_of_member = num_of_client-num_of_head

    head_size=client_size

    member_size=client_size


    #Assigning the indexs of each client
    
    for i in range(num_of_client):
        index_lsit=[]
        for j in range(len(client_data_index[i])):
            index_lsit.append(int(client_data_index[i][j]))
    
        indexs_of_clients.append(index_lsit) 
    #print(len(indexs_of_clients))
    #print(indexs_of_clients[0])
    #print(indexs_of_clients[99])

    #Counting the number of labels of each client
    for i in range(num_of_client):
        num_of_class_list = [0] * label_num
        
        for j in range(len(indexs_of_clients[i])):
            index_of_original_dataset = int(indexs_of_clients[i][j])
            label = train_dataset[index_of_original_dataset][1]
            num_of_class_list[label]=num_of_class_list[label]+1
            
        nums_of_labels_of_each_client.append(num_of_class_list)
        
    #print(len(nums_of_labels_of_each_client))
    #print(nums_of_labels_of_each_client[0])

    #Calculating the KL of each client
    for i in range(num_of_client):
        KL = 0
        for j in range(len(nums_of_labels_of_each_client[i])):
            distribution_P = nums_of_labels_of_each_client[i][j]/(len(train_dataset)/num_of_client)
            
            if distribution_P!=0:
                result_of_division = distribution_P / distribution_Q
                KL = KL + (distribution_P * math.log( result_of_division ,2))
        KL_of_each_client.append(KL)

    #print(len(KL_of_each_client))
    #print(KL_of_each_client)

    ####splitting the head and member####
    #
    head_index = []
    member_index = []
    for i in range(num_of_client):
        member_index.append(i)

    sorted_KL_list = sorted(KL_of_each_client)
    #print(sorted_KL_list)

    #find the heads
    
    for i in range(num_of_head):
        
        for j in range(len(member_index)):
            index_of_member = member_index[j]
            if (KL_of_each_client[index_of_member]==sorted_KL_list[i]) and (index_of_member not in head_index):
                index_of_head = index_of_member
                #print(index_of_head)
                head_index.append(index_of_head)
                member_index.remove(index_of_head)
                break

    #print(len(head_index))
    #print(len(member_index))
    #print(head_index)
    #print(member_index)

    #for i in range(len(head_index)):
        #print(KL_of_each_client[head_index[i]])
    #print("member")
    #for i in range(len(member_index)):
        #print(KL_of_each_client[member_index[i]])


    ####ILP algorithm
    final_size = int((used_client_num*client_size)/num_of_head)
    capacity_of_head = int((final_size-head_size)/member_size)+1
    head_index_group = []
    unassigned_member_index = member_index
    #convert to 2d list
    for i in range(num_of_head):
        index_list=[]
        index_list.append(head_index[i])
        head_index_group.append(index_list)
        
    round_i=0
    #for round_i in range(total_round):
    while (unassigned_member_index):

        #count round i
        round_i=round_i+1
        
        #select head
        selected_head_group = []
        for i in range(num_of_head):
            if len(head_index_group[i]) < capacity_of_head:
                selected_head_group.append(i)
                
        #create the combination of member
        
        Combination_of_C_M = list(itertools.combinations(unassigned_member_index, selected_member_num))
        print(Combination_of_C_M)
        #create KL table
        KL_table = []
        
        for i in range(len(selected_head_group)):

            selected_head_index = selected_head_group[i]
            
            KL_list=[]

            size_of_combination_head = 1 * head_size + (len(head_index_group[selected_head_index])-1) * member_size
            #print(size_of_combination_head)
            P_i_list = [0] * label_num

            for j in range(len(head_index_group[selected_head_index])):
                for t in range(label_num):
                    P_i_list[t] = P_i_list[t] + nums_of_labels_of_each_client[head_index_group[selected_head_index][j]][t]

            
            
            size_of_combination_member = member_size * selected_member_num
            
            for j in range(len(Combination_of_C_M)):

                P_j_list = [0] * label_num

                for k in range(selected_member_num):
                    for t in range(label_num):
                        index_of_member = Combination_of_C_M[j][k]
                        P_j_list[t] = P_j_list[t] + nums_of_labels_of_each_client[index_of_member][t]
            
                KL = KL_of_combination(label_num,size_of_combination_head+size_of_combination_member,P_i_list,P_j_list,distribution_Q)

                KL_list.append(KL)
            
            KL_table.append(KL_list)

            
            #print(P_i_list)
            #print(size_of_combination_member)
            #print(P_j_list)
        #for i in range(num_of_head):
            #print(KL_table[i])

        ###create ILP model
        Max_val = ILP_Max_value
        C_h = len(selected_head_group)
        C_M = len(unassigned_member_index)
        C_M_Combination_num = len(Combination_of_C_M)

        #| Max_val - KL(Pi+Pj,Q) |
        KL_table = np.array(KL_table)
        KL_table = -KL_table + Max_val
        KL_table = abs(KL_table)
        #head_Capacity = [4] * C_h

        #print(head_Capacity)
        #print(KL_table)
        #print(C_h)
        #print(C_M)
        
        #daul varaible
        x = cp.Variable((C_h,C_M_Combination_num), boolean=True)

        # Create constraints.
        constraints = []

        for i in range(C_h):
            constraints.append(sum(x[i][:]) <= 1)
                  
        for j in range(C_M_Combination_num):
            constraints.append(sum(x[:])[j] <= 1)
            
        for j in range(C_M_Combination_num):
            for k in range(j+1,C_M_Combination_num):
                if j!=k and len(set(Combination_of_C_M[j]).intersection(Combination_of_C_M[k]))>0:
                    constraints.append((sum(x[:])[j]+sum(x[:])[k]) <= 1)
                    
        #objective function
        objective = cp.sum(cp.multiply(x,KL_table))
        prob = cp.Problem(cp.Maximize(objective),constraints)


        #solve the problem
        #prob.solve(solver=cp.MOSEK,verbose=True)
        prob.solve(solver=cp.MOSEK)
        print("------------Round", round_i, "------------")
        print("Status: ", prob.status)
        print("The optimal value is", prob.value)
        print("A solution x is")
        print(x.value)
        
        #print(selected_head_group)
        #print(len(selected_head_group),len(unassigned_member_index))
        #assign the member to the head group
        total_assigned_member = 0
        assigned_member_group=x.value
        for i in range(C_h):
            selected_head_group_index = selected_head_group[i]
            
            for j in range(C_M_Combination_num):
                if assigned_member_group[i][j]==1:               
                    print("x(",i,",",j,")=1"," | head :",head_index_group[selected_head_group_index][0]," <= member",Combination_of_C_M[j])
                    total_assigned_member = total_assigned_member + selected_member_num
                    for k in range(selected_member_num):
                        head_index_group[selected_head_group_index].append(Combination_of_C_M[j][k])
                        unassigned_member_index.remove(Combination_of_C_M[j][k])

        if prob.status=="optimal" and total_assigned_member>0:
            print("time: ", prob._compilation_time)
            total_time = total_time + prob._compilation_time       
        #print("total assigned member = ",total_assigned_member)
        #print("unassigned member = ",len(unassigned_member_index))
        for i in range(len(head_index_group)):
            print(head_index_group[i])

        #when round i > threshold ==> stop while loop
        if (round_i>num_of_client) and len(unassigned_member_index)>0:
            num_of_unassigned_member = len(unassigned_member_index)
            unassigned_member_list = unassigned_member_index
            
            while(unassigned_member_index):
                #print(unassigned_member_index)
                
                for i in range(num_of_head):
                    if len(head_index_group[i])<capacity_of_head:
                        index_of_member=unassigned_member_index[0]
                        #print(head_index_group[i],"<=",index_of_member)
                        head_index_group[i].append(index_of_member)
                        unassigned_member_index.remove(index_of_member)
                        break
                    
            print("------------Stop loop------------")
            for i in range(len(head_index_group)):
                print(head_index_group[i])
                
    print(total_time," seconds")    

    #print(unassigned_member_index)
    return head_index_group, total_time

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    train_dataset = datasets.MNIST(
        root="./data/MNIST", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data/MNIST", train=False, download=True, transform=transform
    )
    #print(train_dataset)
    client_num=num_of_original_client
    #print(client_num)
    #divide train dataset(non-iid)

    dict_users = mnist_noniid(train_dataset, client_num)

    ##########heuristic method##########
    print("##########heuristic method##########")
    heuristic_index_of_head_group, time = heuristic_method(train_dataset, num_of_MNIST_label, dict_users, client_num, num_of_used_client, num_of_head_client)
    
    global heuristic_total_execution_time
    heuristic_total_execution_time = heuristic_total_execution_time + time

    #merge train dataset
    num_of_client = num_of_head_client
    sorted_heuristic_method_train_dataset = []
    #print(len(dict_users[0]))
    for k in range(num_of_head_client):
        
        for i in range(len(heuristic_index_of_head_group[k])):
            
            client_index = heuristic_index_of_head_group[k][i]
            #print(client_index)
            for j in range(len(dict_users[client_index])):
                data_index=int(dict_users[client_index][j])
                sorted_heuristic_method_train_dataset.append(train_dataset[data_index])
    heuristic_KL_of_each_client, heuristic_avg_KL = get_KL_value(sorted_heuristic_method_train_dataset, num_of_MNIST_label, num_of_client)
    #print(heuristic_avg_KL)
    ##########ILP method##########
    print("##########ILP method##########")
    ILP_index_of_head_group, ILP_time = ILP_method(train_dataset, num_of_MNIST_label, dict_users, client_num, num_of_used_client, num_of_head_client, Max_value_of_ILP)
    
    global ILP_total_execution_time
    ILP_total_execution_time = ILP_total_execution_time + ILP_time

    #merge train dataset
    num_of_client = num_of_head_client
    sorted_ILP_method_train_dataset = []
    #print(len(dict_users[0]))
    for k in range(num_of_head_client):
         
        for i in range(len(ILP_index_of_head_group[k])):
            client_index = ILP_index_of_head_group[k][i]
            #print(client_index)
            for j in range(len(dict_users[client_index])):
                data_index=int(dict_users[client_index][j])
                sorted_ILP_method_train_dataset.append(train_dataset[data_index])
    ILP_KL_of_each_client, ILP_avg_KL = get_KL_value(sorted_ILP_method_train_dataset, num_of_MNIST_label, num_of_client)
    #print(ILP_avg_KL)
if __name__ == "__main__":
    main()



