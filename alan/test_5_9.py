
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math
import random
from Determined_Client_data import Generate_Client_data
from Determined_Cij import D_Cij
from Determined_Xij import D_Xij


##data

#NCD = 10
#NMD = 10

Client_number = 500
Client_node_number = 10
label = 10
CD = Generate_Client_data(Client_number,label)
MD_label = np.array(CD.CD_label)
MD_label_len = dict(CD.CD_label_len)
node_size = 50
W1 = 0.5
W2 = 0.5
##

Cij = D_Cij(Client_number,Client_node_number,CD,MD_label,MD_label_len,node_size,W1,W2)



print("CD_label:\n",CD.CD_label)
print("CD_label_len:\n",CD.CD_label_len)
print("Cij:",Cij)
#np.savetxt("Cij.csv", Cij,fmt='%1.4f',delimiter = ",")
