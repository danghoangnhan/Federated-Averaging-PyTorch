import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G = nx.erdos_renyi_graph(10,0.5,seed = 123,directed = False)

shotest_sourse = 0
shotest_target = 5



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
shortest_path = nx.shortest_path(G,source=shotest_sourse,target=shotest_target,weight='weight')
path_edges = list(zip(shortest_path,shortest_path[1:]))
distance = nx.shortest_path_length(G,source=shotest_sourse,target=shotest_target,weight='weight')
"""
shortest_path = nx.astar_path(G,source=shotest_sourse,target=shotest_target,weight='weight')
path_edges = list(zip(shortest_path,shortest_path[1:]))
distance = nx.astar_path_length(G,source=shotest_sourse,target=shotest_target,weight='weight')
#print("2",path_edges)
##gernate shortest path on graph

nx.draw_networkx_nodes(G,pos,nodelist=shortest_path,node_color='r')
nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r',width=10)


##draw weight
edge_labels = nx.get_edge_attributes(G, "weight")

#print("edge",edge_labels[path_edges[0]])


nx.draw_networkx_edge_labels(G, pos, edge_labels,font_size = 20,font_color = 'red')
#nx.draw_networkx_edge_labels(G, pos, path_edges,font_size = 20,font_color = 'red')


print("Shotest_path:",shortest_path)
print("Shotest_path_lengh:",distance)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()

"""
print(G.number_of_edges())
nx.draw_networkx(G, pos)
nx.draw_networkx(G)
plt.show()
"""