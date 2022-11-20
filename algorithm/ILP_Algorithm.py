import os
import torch
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
import time
def KL_of_combination(num_of_label,total_dataset_size,P_i,P_j,Q):
    #P_i,P_j => list.
    
    KL = 0
    for i in range(num_of_label):
        P = (P_i[i]+P_j[i])/total_dataset_size
        if P!=0:
            result_of_division = P / Q
            KL = KL + (P * math.log( result_of_division ,2))
    return KL

def ILP_method(train_dataset, num_of_label, client_data_index, original_client_num, head_num, ILP_Max_value):

    total_time = 0
    #client information
    num_of_client = original_client_num

    indexs_of_clients=[]

    nums_of_labels_of_each_client= []

    client_size=len(train_dataset)/num_of_client

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
    final_size = int(len(train_dataset)/num_of_head)
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
            
        #select head
        selected_head_group = []
        for i in range(num_of_head):
            if len(head_index_group[i]) < capacity_of_head:
                selected_head_group.append(i)
                
        #create KL table
        KL_table = []
        round_i=round_i+1
        for i in range(len(selected_head_group)):

            selected_head_index = selected_head_group[i]
            
            KL_list=[]

            size_of_combination_head = 1 * head_size + (len(head_index_group[selected_head_index])-1) * member_size
            
            P_i_list = [0] * label_num

            for j in range(len(head_index_group[selected_head_index])):
                for t in range(label_num):
                    P_i_list[t] = P_i_list[t] + nums_of_labels_of_each_client[head_index_group[selected_head_index][j]][t]
                
            for j in range(len(unassigned_member_index)):
                                          
                P_j_list = nums_of_labels_of_each_client[unassigned_member_index[j]]
            
                KL = KL_of_combination(label_num,size_of_combination_head+member_size,P_i_list,P_j_list,distribution_Q)

                KL_list.append(KL)
            
            KL_table.append(KL_list)
        
        #for i in range(num_of_head):
            #print(KL_table[i])

        ###create ILP model
        Max_val = ILP_Max_value
        C_h = len(selected_head_group)
        C_M = len(unassigned_member_index)
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
        x = cp.Variable((C_h,C_M), boolean=True)

        # Create constraints.
        constraints = []

        for i in range(C_h):
            constraints.append(sum(x[i][:]) <= 1)
                  
        for j in range(C_M):
            constraints.append(sum(x[:])[j] <= 1)
        
        #objective function
        objective = cp.sum(cp.multiply(x,KL_table))
        prob = cp.Problem(cp.Maximize(objective),constraints)


        #solve the problem
        #prob.solve(solver=cp.MOSEK,verbose=True)
        prob.solve(solver=cp.MOSEK)
        print("------------Round", round_i, "------------")
        #print("Status: ", prob.status)
        #print("The optimal value is", prob.value)
        #print("A solution x is")
        #print(x.value)
        
        #print(selected_head_group)
        #print(len(selected_head_group),len(unassigned_member_index))
        #assign the member to the head group
        total_assigned_member = 0
        assigned_member=x.value
        for i in range(C_h):
            selected_head_group_index = selected_head_group[i]
            
            for j in range(len(unassigned_member_index)):
                if assigned_member[i][j]==1:               
                    #print("x(",i,",",j,")=1"," | head :",head_index_group[selected_head_group_index][0]," <= member",unassigned_member_index[j])
                    total_assigned_member = total_assigned_member +1
                    head_index_group[selected_head_group_index].append(unassigned_member_index[j])
                    unassigned_member_index.remove(unassigned_member_index[j])

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
                
    print(total_time," second")    

    #print(unassigned_member_index)
    return head_index_group, total_time


  


    



