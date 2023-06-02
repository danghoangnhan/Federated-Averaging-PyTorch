
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math
import random
from Determined_KLD import FKLD
from Determined_network import Generate_network
from Determined_Client import Generate_Client

##data
NCD = 10
NMD = 10
Client_number = 500
Client_node_number = 10
node_size = 50
max_edge_weight = 10
label = 10
Cij = np.zeros([NCD,NMD])

##  generate Client
Client_node,Client_arr = Generate_Client(Client_number,Client_node_number)
##  generate CD
CD_data = dict()
CD_label = np.zeros([NCD,label])
CD_label_len = {}

### CD_label_le & MD_label_len 是一維陣列

### CD_label & MD_label是 每個label在 client 內的個數

for i in range(NCD):
        data_size = random.randint(10,100)
        CD_data[i] = np.random.randint(0,10,size=data_size)
        CD_data[i] = sorted(CD_data[i])
for i in range(NCD):
    temp = CD_data[i]
    CD_label_len[i] = len(temp)
    for j in range(len(temp)):
        CD_label[i][temp[j]] = CD_label[i][temp[j]] + 1
## MD
MD_label = CD_label
MD_label_len = CD_label_len
print("CD_label: ",CD_label)
KL = FKLD(NCD,NMD,CD_label,CD_label_len,MD_label,MD_label_len)
###算Mcij
W1 = 0.5
W2 = 0.5
### 產生network
shortest_path,shortest_path_length = Generate_network(node_size,max_edge_weight,Client_node)
print("shortest_path: ",shortest_path)
print("distance: ",shortest_path_length)
###正規化
normal_tran = dict(shortest_path_length)
max_trans = -1
min_trans = max_edge_weight + 1

normal_KL = np.array(KL)
max_KL = np.nanmax(KL)
min_KL = np.nanmin(KL)

for value in shortest_path_length.values():
     temp_value = value
     print("Value: ",value)
     for j in range(len(temp_value)):
        if(temp_value[j] != 0):
            max_trans = max(max_trans,temp_value[j])
            min_trans = min(min_trans,temp_value[j])

print("shortest_path_length: ",shortest_path_length)
for i in range(node_size):
     for j in range(label):
            if(i != j):
                normal_tran[i][j] = (normal_tran[i][j] - min_trans) / (max_trans - min_trans)
#print("normal_tran: ",normal_tran)  

for i in range(node_size):
     for j in range(label):
        if(normal_KL[i][j] != np.nan):
            normal_KL[i][j] = (normal_KL[i][j] - min_KL) / (max_KL - min_KL)
#print("normal_KL:", normal_KL)     

         
print("KL: ",KL)
print("max_KL: ",max_KL)
print("min_KL: ",min_KL)
print("max_trans: ",max_trans)
print("min_trans: ",min_trans)


"""
print("shortest_path: ",shortest_path)
print("shortest_path_length: ",shortest_path_length)
ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()
"""