
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math
import random
from Determined_Client import Generate_Client
from Determined_Cij import D_Cij
from Determined_Xij import D_Xij
#from Determined_Xij_copy import D_Xij


##data

#NCD = 10
#NMD = 10

Client_number = 100
Client_node_number = 10
label = 10
CD = Generate_Client(Client_number,Client_node_number,label)
print("CD_label:\n",CD.label)
print("CD_label_len:\n",CD.label_len)

MD_label = np.array(CD.label)
MD_label_len = dict(CD.label_len)

print("MD_label:\n",MD_label)
print("MD_label_len:\n",MD_label_len)
''''''
node_size = 50
W1 = 0.5
W2 = 0.5
##

Cij,not_comp = D_Cij(Client_number,Client_node_number,CD,MD_label,MD_label_len,node_size,W1,W2)
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
"""
#print("not_com:",len(not_comp))
#print("Cij:",Cij)
Xij=np.array(D_Xij(Client_number,len(not_comp),Cij))
print("test:\n",Xij) 
np.savetxt("Xij.csv", Xij,fmt='%1.4f',delimiter = ",")
check = 0
for i in range(np.size(Xij,0)):
    for j in range(np.size(Xij,1)):
        check = check + Xij[i][j]
print("not_com:",not_comp)       
#print("CD_label:\n",CD.label)
#print("CD_label_len:\n",CD.label_len)
print("Cij:",Cij)
print("check :",check)
#np.savetxt("Cij.csv", Cij,fmt='%1.4f',delimiter = ",")
