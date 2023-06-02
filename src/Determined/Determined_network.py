import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def Generate_network(node_size,min_edge_weight,max_edge_weight,Client_node):

    G = nx.erdos_renyi_graph(node_size,0.5,directed = False)

    
    ## generate edge
    for u,v,d in G.edges(data=True):
        d['weight'] =  np.random.choice(np.arange(min_edge_weight, max_edge_weight))
    for shortest in G.nodes:
        G.nodes[shortest]['shortest'] = False

    #If you want to print out the edge weights:
    labels = nx.get_edge_attributes(G,'weight')
    #test = nx.get_node_attributes(G,'shortest')
    #print("Here are the edge weights: ", labels)
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
    ## Astar
    """
    shortest_path = nx.astar_path(G,source=shotest_source,target=shotest_target,weight='weight')
    path_edges = list(zip(shortest_path,shortest_path[1:]))
    distance = nx.astar_path_length(G,source=shotest_source,target=shotest_target,weight='weight')
    """
    #print("2",path_edges)
    ##gernate shortest path on graph
    """
    nx.draw_networkx_nodes(G,pos,nodelist=shortest_path,node_color='r')
    nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='r',width=10)
    """

    ##draw weight
    edge_labels = nx.get_edge_attributes(G, "weight")

    #print("edge",edge_labels[path_edges[0]])


    nx.draw_networkx_edge_labels(G, pos, edge_labels,font_size = 20,font_color = 'red')
    #nx.draw_networkx_edge_labels(G, pos, path_edges,font_size = 20,font_color = 'red')

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
    #5/8 shortest path
    shortest_path = dict()
    #distance = np.zeros([len(Client_node),len(Client_node)])
    distance = dict()
    """舊的流下來
    for i in range(len(Client_node)):
        for j in range(len(Client_node)):
            if(not isself):
                if(i != j):
                    #print("not isself",not isself)
                    shortest_path[(Client_node[i],Client_node[j])] = nx.shortest_path(G,source=Client_node[i],target=Client_node[j],weight='weight')
                    #path_edges = list(zip(shortest_path[(i,j)],shortest_path[1:]))
                    distance[(Client_node[i],Client_node[j])] = nx.shortest_path_length(G,source=Client_node[i],target=Client_node[j],weight='weight')
            else:
                #print("isself",isself)
                shortest_path[(Client_node[i],Client_node[j])] = nx.shortest_path(G,source=Client_node[i],target=Client_node[j],weight='weight')
                #path_edges = list(zip(shortest_path[(i,j)],shortest_path[1:]))
                distance[(Client_node[i],Client_node[j])] = nx.shortest_path_length(G,source=Client_node[i],target=Client_node[j],weight='weight')
    """ 
    for i in range(len(Client_node)):
        for j in range(len(Client_node)):
            if(i != j):
                shortest_path[(Client_node[i],Client_node[j])] = nx.shortest_path(G,source=Client_node[i],target=Client_node[j],weight='weight')
                #path_edges = list(zip(shortest_path[(i,j)],shortest_path[1:]))
                distance[(Client_node[i],Client_node[j])] = nx.shortest_path_length(G,source=Client_node[i],target=Client_node[j],weight='weight')
         
    """
    print("shortest_path: ",shortest_path)
    print("distance: ",distance)
    """
    return shortest_path,distance
    


    '''
    print("single path 0:",nx.single_source_shortest_path(G,0))
    path = dict(nx.all_pairs_shortest_path(G))
    print("all path single path 0:",path[0][5])
    '''
    """
    #dijkstra
    dijkstra_path = dict(nx.all_pairs_dijkstra_path(G,weight='weight'))
    dijkstra_path_length = dict(nx.all_pairs_dijkstra_path_length(G,weight='weight'))
    """
    """
    print("dijkstra_all_path:",dijkstra_path)
    print("dijkstra_all_path_length:",dijkstra_path_length)
    print("dijkstra_all_path 0-5:",dijkstra_path[0][5])
    print("dijkstra_all_path_length 0-5:",dijkstra_path_length[0][5])
    
    
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    """
    return dijkstra_path,dijkstra_path_length

    """
    print("Shotest_path:",shortest_path)
    print("Shotest_path_lengh:",distance)

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
    