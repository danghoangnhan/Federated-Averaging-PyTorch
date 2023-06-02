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
def Orign_Client_KLD(CD_label,CD_label_len,label,Client_number):
    Q = 1 / label
    for i in range(np.size(CD_label,axis=0)):
        not_comp_temp = []
        total_Client_label_p = np.zeros(Client_number)
        for j in range(np.size(CD_label,axis=1)):
            
            temp_p = CD_label[i][j] / CD_label_len[i]
            print("temp_p:",temp_p)
            if(temp_p != 0):
                total_Client_label_p[i] = total_Client_label_p[i] + (temp_p * math.log(temp_p / Q ,2))
            
        
    print("total_Client_label_p:",total_Client_label_p)
    #print("not_comp:",not_comp)
    return sum(total_Client_label_p) / len(total_Client_label_p)

def Client_KLD(MD_label,MD_label_len,label,avg_orign_KL):
    Q = 1 / label
    #Q = Q / 2
    not_comp = []
    comp = []
    have_all_label = False
    for i in range(np.size(MD_label,axis=0)):
        not_comp_temp = []
        total_label_p = 0
        for j in range(np.size(MD_label,axis=1)):
            temp_p = MD_label[i][j] / MD_label_len[i]
            #print("temp_p:",temp_p)
            if(temp_p != 0):
                total_label_p = total_label_p + (temp_p * math.log(temp_p / Q ,2))
        #print("total_label_p[",i,"]:",total_label_p)
        if(total_label_p > avg_orign_KL):
            not_comp.append(i)
        else:
            comp.append(i)
    
    #print("not_comp:",not_comp)
    return not_comp,comp
def Check_all_label(MD_label,MD_label_len,label):
    Q = 1 / label
    #Q = Q / 2
    not_comp = []
    comp = []
    
    for i in range(np.size(MD_label,axis=0)):
        have_all_label = True
        total_label_p = 0
        for j in range(np.size(MD_label,axis=1)):
            if(MD_label[i][j] == 0):
                have_all_label = False
            temp_p = MD_label[i][j] / MD_label_len[i]
            #print("temp_p:",temp_p)
            if(temp_p != 0):
                total_label_p = total_label_p + (temp_p * math.log(temp_p / Q ,2))
        #print("total_label_p[",i,"]:",total_label_p)
        if(have_all_label):
            comp.append(i)
        else:
            not_comp.append(i)
    
    #print("not_comp:",not_comp)
    return not_comp,comp
#0.008393