import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

node_size = 50
Client_number = 500
Client_node_numebr = 10
G = nx.erdos_renyi_graph(node_size,0.5,seed = 123,directed = False)

Client_arr = np.zeros([Client_number])


lst = list(np.arange(0,50))
random.shuffle(lst)
Client_node=lst[0:Client_node_numebr]
Client_node = sorted(Client_node)
for i in range(len(Client_node)):
    Client_arr[i] = Client_node[i]
for i in range(Client_node_numebr,Client_number):
    Client_arr[i] = Client_node[random.randint(0,len(Client_node) - 1)]
print(lst)
print(Client_node)
print(Client_arr)


## generate edge
for u,v,d in G.edges(data=True):
    d['weight'] =  np.random.choice(np.arange(1, 7))
for shortest in G.nodes:
    G.nodes[shortest]['shortest'] = False

#If you want to print out the edge weights:
labels = nx.get_edge_attributes(G,'weight')
#test = nx.get_node_attributes(G,'shortest')
print("Here are the edge weights: ", labels)
# print("Here are the edge shortest: ", test)



pos = nx.shell_layout(G)
#pos = nx.spring_layout(G,k=0.05)
nx.draw_networkx(G, pos)

##Find shortest path
"""


shortest_path = nx.astar_path(G,source=shotest_source,target=shotest_target,weight='weight')
path_edges = list(zip(shortest_path,shortest_path[1:]))
distance = nx.astar_path_length(G,source=shotest_source,target=shotest_target,weight='weight')
#print("2",path_edges)
##gernate shortest path on graph

nx.draw_networkx_nodes(G,pos,nodelist=shortest_path,node_color='r')
nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r',width=10)


##draw weight
edge_labels = nx.get_edge_attributes(G, "weight")

#print("edge",edge_labels[path_edges[0]])


nx.draw_networkx_edge_labels(G, pos, edge_labels,font_size = 20,font_color = 'red')
#nx.draw_networkx_edge_labels(G, pos, path_edges,font_size = 20,font_color = 'red')
"""
##4/28 all shortest path
'''
all_shortest_path_list = []
for source in range(node_size):
    for target in range(node_size):
        all_shortest_path_list.append(nx.astar_path(G,source=source,target=target,weight='weight'))

print("all path list:",all_shortest_path_list)
'''
#dictionary
''''
path = dict
for source in range(node_size):
    for target in range(node_size):
        path[source][target] = nx.astar_path(G,source=source,target=target,weight='weight')
for source in range(node_size):
    for target in range(node_size):
        print("source: ",source,"target: ",target,"path: ",path[source][target])

'''
shortest_path = dict()
distance = np.zeros([Client_node_numebr,Client_node_numebr])
for i in range(len(Client_node)):
     for j in range(len(Client_node)):
        if(i != j):
            shortest_path[(Client_node[i],Client_node[j])] = nx.shortest_path(G,source=Client_node[i],target=Client_node[j],weight='weight')
            #path_edges = list(zip(shortest_path[(i,j)],shortest_path[1:]))
            distance[i][j] = nx.shortest_path_length(G,source=Client_node[i],target=Client_node[j],weight='weight')
        else:
            distance[i][j] = np.nan
print("shortest_path: ",shortest_path)
print("distance: ",distance)



'''
print("single path 0:",nx.single_source_shortest_path(G,0))
path = dict(nx.all_pairs_shortest_path(G))
print("all path single path 0:",path[0][5])
'''
#dijkstra
"""
dijkstra_path = dict(nx.all_pairs_dijkstra_path(G,weight='weight'))
dijkstra_path_length = dict(nx.all_pairs_dijkstra_path_length(G,weight='weight'))
print("dijkstra_all_path:",dijkstra_path)
print("dijkstra_all_path_length:",dijkstra_path_length)
print("dijkstra_all_path 0-5:",dijkstra_path[0][5])
print("dijkstra_all_path_length 0-5:",dijkstra_path_length[0][5])



print("Shotest_path:",shortest_path)
print("Shotest_path_lengh:",distance)
"""
"""
ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()
"""
"""
print(G.number_of_edges())
nx.draw_networkx(G, pos)
nx.draw_networkx(G)
plt.show()
"""