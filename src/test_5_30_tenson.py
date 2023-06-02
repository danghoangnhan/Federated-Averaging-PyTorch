
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math
import random
import copy
from Determined.Determined_Client import Generate_Client,Generate_Client_to_CD
from Determined.Determined_Cij import D_Cij
from Determined.Determined_Xij import D_Xij
from Determined.Determined_Client_KLD import Client_KLD,Orign_Client_KLD
from Determined.Determined_network import Generate_network

###test
from label import label_group
from utils import get_dataset, average_weights, exp_details
from options import args_parser
from torchvision import datasets, transforms
#from Determined_Xij_copy import D_Xij


##data

#NCD = 10
#NMD = 10
"""
Client_number = 100
Client_node_number = 10
label = 10
node_size = 50
"""
Client_number = 20
Client_node_number = 5
label = 10
node_size = 10
W1 = 0.5
W2 = 0.5
min_edge_weight = 5
max_edge_weight = 10
isself = True 
used_Client =dict()
for i in range(Client_number):
    used_Client[i] = []
print("used_Client :",used_Client )
###test
data_dir = '../data/mnist/'
apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)


origin = [train_dataset[i] for i in range(len(train_dataset))]
label_per_group, labelcount = label_group(
            sorted_train_dataset=origin,
            groupSize=100
            )
###
#CD = Generate_Client(Client_number,node_size,Client_node_number,label)
CD = Generate_Client_to_CD(Client_number,node_size,Client_node_number,label,labelcount)
print("CD_label:\n",CD.label)
print("CD_label_len:\n",CD.label_len)

MD_label = np.array(CD.label)
MD_label_len = np.array(CD.label_len)

print("MD_label:\n",MD_label)
print("MD_label_len:\n",MD_label_len)
''''''
print("CD.node:",CD.node)
shortest_path,shortest_path_length = Generate_network(node_size,min_edge_weight,max_edge_weight,CD.node,isself)
Client_modle = np.zeros([Client_number,2],int)
for i in range(Client_number):
    Client_modle[i][0] = i
    Client_modle[i][1] = -1
##

avg_orign_KL = Orign_Client_KLD(CD.label,CD.label_len,label,Client_number)
print("avg_orign_KL:",avg_orign_KL)
not_comp,comp = Client_KLD(MD_label,MD_label_len,label,avg_orign_KL)
count = 0
for round in range(50):
#while len(not_comp) != 0:
    count = count + 1
    not_comp = []
    comp = []
    not_comp,comp = Client_KLD(MD_label,MD_label_len,label,avg_orign_KL)


    C_ij= D_Cij(Client_number,Client_node_number,CD,MD_label,MD_label_len,node_size,W1,W2,shortest_path_length,comp,isself)
    """
    GL = np.arange(0,10)
    OL = CD.label
    TL = np.array(OL)
    LL =np.array(TL)

    for i in range(np.size(TL,0)):
        for j in range(np.size(TL,1)):
            if(TL[i][j] == 1):
                LL[i][j] = 0
            else:
                LL[i][j] = 1
    n_L = np.array(OL@LL.T)
    for i in range(np.size(n_L,0)):
        for j in range(np.size(n_L,1)):
            n_L[i][j] = np.size(GL)- n_L[i][j]  

    print("GL:\n",GL)
    print("OL:\n",OL)
    print("TL:\n",TL)
    print("LL:\n",LL)
    print("n_L:\n",n_L)
[ 0 14]
 [ 1  5]
 [ 2 10]
 [ 3  6]
 [ 4 13]
 [ 5  8]
 [ 6  9]
 [ 7 17]
 [ 8  1]
 [ 9  3]
 [10  2]
 [11 12]
 [12 18]
 [13  7]
 [14  0]
 [15 16]
 [16 19]
 [17  4]
 [18 15]
 [19 11]]
 358
  [[ 1.  2.  4.  0.  1.  0.  1.  3.  0.  3.]
 [ 3.  2.  1.  2.  3.  1.  0.  0.  1.  3.]
    """
    #print("not_com:",len(not_comp))
    #print("Cij:",Cij)
    X_ij=np.array(D_Xij(Client_number,len(not_comp),C_ij))
    print("not_comp:",not_comp) 
    not_comp.clear()
    #print("test:\n",X_ij) 
    #np.savetxt("Xij.csv", Xij,fmt='%1.4f',delimiter = ",")
    """
    check = 0
    for i in range(np.size(Xij,0)):
        for j in range(np.size(Xij,1)):
            check = check + Xij[i][j]
    print("not_com:",not_comp)
    """    
      
    #print("CD_label:\n",CD.label)
    #print("CD_label_len:\n",CD.label_len)
    #print("Cij:",C_ij)
    #print("check :",check)
    #np.savetxt("Cij.csv", Cij,fmt='%1.4f',delimiter = ",")
    for i in range(Client_number):
        Client_modle[i][1] = -1
        for i in range(np.size(X_ij,0)):
            for j in range(np.size(X_ij,1)):
                if(X_ij[i][j] >= 0.5 ):
                    X_ij[i][j] = 1
                else:
                    X_ij[i][j] = 0
                ##if(X_ij[i][j] != 0 ):
                ##    print("not i:",i," j:",j)
                ##check = check + X_ij[i][j]
                if(X_ij[i][j] != 0):
                    Client_modle[i][1] = j
    print("Client_modle:",Client_modle)
    temp_label = copy.deepcopy(MD_label)
    temp_label_len = copy.deepcopy(MD_label_len)
    for i in range(Client_number):
        if(Client_modle[i][1] != -1):
            for j in range(label):
                MD_label[i][j] = MD_label[i][j] + temp_label[Client_modle[i][1]][j]
                #print("Client_modle[",i,"][1]:",Client_modle[i][1])
                #print("temp_label_len[Client_modle[",i,"][1]:",temp_label_len[Client_modle[i][1]])
            MD_label_len[i] = MD_label_len[i] + temp_label_len[Client_modle[i][1]]
        else:
            for j in range(label):
                MD_label[i][j] = MD_label[i][j] + temp_label[i][j]
            MD_label_len[i] = MD_label_len[i] + temp_label_len[i]
    print("MD_label:",MD_label)
    print("MD_label_len:",MD_label_len)
print("count:",count)
#0~18
 #[[ 1.  2.  6.  0.  4.  2.  3.  0.  4.  2.]
# [10.  9.  3.  5.  5.  6.  6.  8.  5.  3.]
#[11. 11.  9.  5.  9.  8.  9.  8.  9.  5.]
#6266348