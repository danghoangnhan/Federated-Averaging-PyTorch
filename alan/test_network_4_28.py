import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import csv
import sys
sys.path.append("c:\\Users\\mobile_lab\\Desktop\\408261204\\Federated-Averaging-PyTorch-main")

from script.getKL import get_KL_value
from torchvision import datasets, transforms


#data
#data_size = random.randint(10,100)
label = 10

CD_label_len = {}
total = {}
NCD = 10
NMD = 10
CD_data = dict()
CD_label = np.zeros([NCD,label])
#code
#print(CD_data) 
###generate random labels
for i in range(NCD):
    data_size = random.randint(10,100)
    CD_data[i] = np.random.randint(0,10,size=data_size)
    CD_data[i] = sorted(CD_data[i])
#print(CD_data)
###count CD_data in CD_label
for i in range(NCD):
    temp = CD_data[i]
    CD_label_len[i] = len(temp)
    for j in range(len(temp)):
        CD_label[i][temp[j]] = CD_label[i][temp[j]] + 1
        #print(CD)
#print("CD_label[0]: ",CD_label[0])
### old md/cd
"""
for i in range(label):
    temp = CD_data[i]
    for j in range(label):
        CD_label[i][j] = CD_label[i][j] / len(temp) 
""" 
#print("CD_label[0]: ",CD_label[0])
total = np.sum(CD_label,axis=1)
#print("total[0]",total[0])
           


"""
for i in range(len(CD_data[i])):
    CD[CD_data[i] - 1] = CD[CD_data[i] - 1] + 1

for i in range(len(CD)):
    CD[i] = CD[i] / data_size
for i in range(len(CD)):
    total = total + CD[i]
"""
'''
print("CD_data:",CD_data) 
print("CD_label: ",CD_label)
print("CD_label_len: ",CD_label_len)
print(len(CD_data))  
#print(total)
'''
### MD
MD_label = CD_label
MD_label_len = CD_label_len
MD_CD = np.array([NCD,NMD])
MD_CD_total_label = dict()
#print(MD_label[0])
MD_CD_total_label_P = dict()
for i in range(NCD):
    for j in range(NMD):
         if(i != j):
            MD_CD_total_label[(i,j)] = MD_label[i] + CD_label[j]
         
#print(" MD_CD_total_label[(i,j)]:",MD_CD_total_label)
#print(" MD_CD_total_label[(i,j)] p:",MD_CD_total_label[(0,1)])

for i in range(NCD):
    for j in range(NMD):   
        if(i != j):         
            MD_CD_total_label_P[(i,j)] = MD_CD_total_label[(i,j)] /(MD_label_len[i]+CD_label_len[j])


"""
##check first total p != 1
temp_total = 0
temp = MD_CD_total_label_P[(0,0)]
for i in range(label):
    temp_total = temp_total + temp[i]
print("temp: ",temp)
print("temp_total: ",temp_total)
"""

""""""
print("MD_label: ",MD_label)
print("MD_label_len: ",MD_label_len)
print("MD_CD_total_label_P: ",MD_CD_total_label_P)

###test mnist
"""
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
dataset1 = datasets.MNIST(root='./data/', train=True, download=True,
                              transform=transform)
print("dataset: ",dataset1)
KL_of_each_client, avg_KL = get_KL_value(dataset1, 10, 1)
print("KL: ",KL_of_each_client)
print("avg_KL: ",avg_KL)
"""
"""
KL = 0
    for i in range(num_of_label):
        P = P_i[i] / dataset_size
        if P != 0:
            result_of_division = P / Q
            KL = KL + (P * math.log(result_of_division, 2))
    return KL
"""
KL = np.zeros([NCD,NMD])
Q = 1 / label
print("Q: ",Q)
for i in range(NCD):
    for j in range(NMD):
        if(i != j):
            temp_label = MD_CD_total_label_P[(i,j)]
            for p in range(label):
                if temp_label[p] != 0:
                    KL[i][j] = KL[i][j] + (temp_label[p] * math.log(temp_label[p] / Q ,2))
print("KL:",KL)

#np.savetxt("data3.csv", KL,fmt='%1.4f',delimiter = ",")



