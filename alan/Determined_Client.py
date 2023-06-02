import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def Generate_Client(Client_number,Client_node_number):

    Client_arr = np.zeros([Client_number])


    lst = list(np.arange(0,50))
    random.shuffle(lst)
    Client_node=lst[0:Client_node_number]
    Client_node = sorted(Client_node)
    for i in range(len(Client_node)):
        Client_arr[i] = Client_node[i]
    for i in range(Client_node_number,Client_number):
        Client_arr[i] = Client_node[random.randint(0,len(Client_node) - 1)]
    
    print(lst)
    print(Client_node)
    print(Client_arr)
    

    return Client_node,Client_arr
