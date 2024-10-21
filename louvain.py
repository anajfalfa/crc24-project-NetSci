#Louvain Alogorithm

import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from community import community_louvain
import time
import aux_functions as aux

# Main function
if __name__ == "__main__":
    init_time = time.time()

    np.random.seed(42)  # Set seed for NumPy
    # Read the file and build the graph
    graphs = {}
    with open('./facebook_combined.txt') as f:
        G = nx.Graph()
        for line in f:
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    
    method = 'only_combined_txt'# "w_facebook_dir_info" # | "only_combined_txt"
    case = '_simple'

    if method == "w_facebook_dir_info":
        case = '_egdes_subnet_info'

        # 10 sub-networks
        for i in [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]:
        #for i in [0, 107]:
            edges = []
            feat = []
            featnames = []
            
            with open('facebook/{}.edges'.format(i)) as f:
                for line in f:
                    u, v = map(int, line.strip().split())
                    edges.append((u, v))
            
            with open('facebook/{}.feat'.format(i)) as f:
                for line in f:
                    row = list(map(int, line.strip().split()))
                    feat.append(row)
                # add featnames
                with open('facebook/{}.featnames'.format(i)) as f:
                    for line in f:
                        featnames.append(line.strip())
            #Builf sub-graph            
            G_i = nx.Graph()
            G_i.add_edges_from(edges)

            #Create a dictionary of node features
            node_feat_dict = {i+1:feat[i] for i in range(len(feat))}
            # set node attributes
            nx.set_node_attributes(G_i, node_feat_dict, 'feat')
            graphs[i] = G_i

            # Conducting subgraph community partition and calculate the modularity
            partition = community_louvain.best_partition(graphs[i])
            modularity = community_louvain.modularity(partition, graphs[i])
            print(f'Modularity of subnetwork {i} ：', modularity)
            
            # Plotting graph of sub-networks 
            plt.figure(figsize=(10, 10))
            pos = nx.spring_layout(graphs[i], seed=42)
            cmap = plt.get_cmap('tab20')
            colors = [mcolors.to_rgba(cmap(partition[i])) for i in graphs[i].nodes()]
            nx.draw_networkx_nodes(graphs[i], pos, node_size=50, node_color=colors)
            nx.draw_networkx_edges(graphs[i], pos, alpha=0.5)
            plt.title(f'Sub-network {i} Community Detection (Louvain)', fontsize=16)

            plt.axis('off')
            plt.savefig(f'./images/subnetwork_{i}_louvain.png')
            plt.show()
            
            ##Print every community in the subnetwork
            #for com in set(partition.values()):
            #    print("Community %d:" % com)
            #    members = [nodes for nodes in partition.keys() if partition[nodes] == com]
            #    print(members)
            
            # Combined the sub-networks
            G.add_edges_from(edges)

        '''# Degrees of Separation
        time1 = time.time()
        # Calcular o comprimento médio dos caminhos mais curtos na rede completa
        if nx.is_connected(G):
            average_shortest_path = nx.average_shortest_path_length(G)
            print(f"A média de graus de separação (comprimento do caminho mais curto) é: {average_shortest_path}")
            # A média de graus de separação (comprimento do caminho mais curto) é: 3.6925068496963913
        else:
            print("O grafo completo não está totalmente conectado. Existem componentes desconexos.")
        time2 = time.time()
        print("Time (s)",time2-time1)'''

    save_fig_name = './images/' + 'full_network_louvain'+case+'.png'
    partition = community_louvain.best_partition(G)

    algorithm_time = time.time()
    aux.save_partition_with_pickle(partition, "partition_louvain.pkl")

    print("Number of communities:", len(set(partition.values())))

    print("Time (s):", algorithm_time-init_time)

    # print(partition)
    modularity = community_louvain.modularity(partition, G)
    print('Modularity of the full network ：', modularity)
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, seed=42)

    # Plot the complete network graph
    if method == "w_facebook_dir_info":

        for i in [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]:
        #for i in [0, 107]:
            partition = community_louvain.best_partition(graphs[i])
            cmap = plt.get_cmap('tab20')
            colors = [mcolors.to_rgba(cmap(partition[j])) for j in graphs[i].nodes()]
            nx.draw_networkx_nodes(graphs[i], pos, node_size=50, node_color=colors)
            nx.draw_networkx_edges(graphs[i], pos, alpha=0.5)
    
    if method == "only_combined_txt":
        community_colors = [partition[node] for node in G.nodes()]
        cmap =plt.get_cmap('tab20')
        pos = nx.spring_layout(G, seed=42)

        unique_communities = set(partition.values())
        colors = {comm: mcolors.to_rgba(cmap(i)) for i, comm in enumerate(unique_communities)}

        for node in G.nodes():
            community_id = partition[node]
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=10, node_color=[colors[community_id]])

        nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='lightgrey')

    # plt.title(f'Full Network Community Detection (Louvain)', fontsize=16)
    plt.axis('off')
    plt.savefig(save_fig_name, dpi=300)
    plt.show()

    cluster_files = ["facebook/0.edges", 
                     "facebook/107.edges", 
                     "facebook/348.edges", 
                     "facebook/414.edges", 
                     "facebook/686.edges", 
                     "facebook/698.edges", 
                     "facebook/1684.edges", 
                     "facebook/1912.edges", 
                     "facebook/3437.edges", 
                     "facebook/3980.edges"]
    
    # Load the partition later
    loaded_partition = aux.load_partition_with_pickle("partition_louvain.pkl")

    # Load the ground truth clusters
    ground_truth = aux.load_ground_truth(cluster_files)

    # Evaluate clustering performance
    nmi, ari = aux.evaluate_clustering(loaded_partition, ground_truth, 'Louvain')
    print(f"Normalized Mutual Information (NMI): {nmi}")
    print(f"Adjusted Rand Index (ARI): {ari}")
    final_time = time.time()

    print("Time (s):", final_time-init_time)