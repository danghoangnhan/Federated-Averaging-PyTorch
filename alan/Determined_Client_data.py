
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import math
import random
from Determined_KLD import FKLD
from Determined_network import Generate_network
from Determined_Client import Generate_Client
class Client:
    def __init__(self, data, label,label_len):
        self.CD_data = data
        self.CD_label = label
        self.CD_label_len = label_len

     

def Generate_Client_data(Client_number,label):

    ##data
    #NCD = 10
    #NMD = 10
    """
    Client_number = 500
    Client_node_number = 10
    node_size = 50
    label = 10
    """
    CD = Client(dict(),np.zeros([Client_number,label]),{})

    
    ##  generate CD
    """
    CD_data = dict()

    CD_label = np.zeros([Client_number,label])
    CD_label_len = {}
    """

    ### CD_label_le & MD_label_len 是一維陣列

    ### CD_label & MD_label是 每個label在 client 內的個數

    for i in range(Client_number):
            data_size = random.randint(10,100)
            CD.CD_data[i] = np.random.randint(0,10,size=data_size)
            CD.CD_data[i] = sorted(CD.CD_data[i])
    for i in range(Client_number):
        temp = CD.CD_data[i]
        CD.CD_label_len[i] = len(temp)
        for j in range(len(temp)):
            CD.CD_label[i][temp[j]] = CD.CD_label[i][temp[j]] + 1
    return CD
    