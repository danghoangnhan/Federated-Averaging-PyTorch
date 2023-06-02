import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

class Client:
    def __init__(self, data, label,label_len,Client_node,Client_arr):
        self.data = data
        self.label = label
        self.label_len = label_len
        self.node = Client_node
        self.arr = Client_arr


def Generate_Client(Client_number,node_size,Client_node_number,label):
    ##Client data
    CD = Client(dict(),np.zeros([Client_number,label]),{},{},np.zeros([Client_number]))

    
    ##  generate CD

    ### CD_label_len & MD_label_len 是一維陣列

    ### CD_label & MD_label是 每個label在 client 內的個數

    for i in range(Client_number):
            data_size = random.randint(10,100)
            CD.data[i] = np.random.randint(0,10,size=data_size)
            CD.data[i] = sorted(CD.data[i])
    for i in range(Client_number):
        temp = CD.data[i]
        CD.label_len[i] = len(temp)
        for j in range(len(temp)):
            CD.label[i][temp[j]] = CD.label[i][temp[j]] + 1
    ##Client node
    ###
    """
    Client_node 是50個node中選出10個
    Client arr 是500個client在那些node上
    """
    #Client_arr = np.zeros([Client_number])
    

    lst = list(np.arange(0,node_size))
    random.shuffle(lst)
    CD.node=lst[0:Client_node_number]
    CD.node= sorted(CD.node)
    for i in range(len(CD.node)):
        CD.arr[i] = CD.node[i]
    for i in range(Client_node_number,Client_number):
        CD.arr[i] = CD.node[random.randint(0,len(CD.node) - 1)]
    
    """
    print(lst)
    print("CD.node:",CD.node)
    print("CD.arr:",CD.arr)
    """

    return CD
def Generate_Client_to_CD(Client_number,node_size,Client_node_number,label,train_label):
    ##Client data
    CD = Client(dict(),np.zeros([Client_number,label]),np.zeros([Client_number]),{},np.zeros([Client_number]))

    
    ##  generate CD

    ### CD_label_len & MD_label_len 是一維陣列

    ### CD_label & MD_label是 每個label在 client 內的個數
    """ random 產生client data
    for i in range(Client_number):
            data_size = random.randint(10,100)
            CD.data[i] = np.random.randint(0,10,size=data_size)
            CD.data[i] = sorted(CD.data[i])
    for i in range(Client_number):
        temp = CD.data[i]
        CD.label_len[i] = len(temp)
        for j in range(len(temp)):
            CD.label[i][temp[j]] = CD.label[i][temp[j]] + 1
    """

    for i in range(Client_number):
        temp = train_label[i]
        #print("temp",temp)
        #print("temp[0]",temp[0])
        for j in range(label):
            CD.label[i][j] = temp[j] 
            CD.label_len[i] = CD.label_len[i] + temp[j]
    """
    print("CD.label:",CD.label[0][0])
    """
    print("CD.label:",CD.label)
    print("CD.label_len:",CD.label_len)
    
    ##Client node
    ###
    """
    Client_node 是50個node中選出10個
    Client arr 是500個client在那些node上
    """
    #Client_arr = np.zeros([Client_number])
    

    lst = list(np.arange(0,node_size))
    random.shuffle(lst)
    CD.node=lst[0:Client_node_number]
    CD.node= sorted(CD.node)
    for i in range(len(CD.node)):
        CD.arr[i] = CD.node[i]
    for i in range(Client_node_number,Client_number):
        CD.arr[i] = CD.node[random.randint(0,len(CD.node) - 1)]
    
    """
    print(lst)
    print("CD.node:",CD.node)
    print("CD.arr:",CD.arr)
    """

    return CD
