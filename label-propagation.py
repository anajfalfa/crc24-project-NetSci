# Label Propagation Algorithm

import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.algorithms.community import label_propagation_communities
import time
import aux_functions as aux
from community import community_louvain

# Main function
if __name__ == "__main__":
    init_time = time.time()
    np.random.seed(42)  # Set seed for NumPy

    graphs = {}
    with open('./facebook_combined.txt') as f:
        G = nx.Graph()
        for line in f:
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    
    method = "only_combined_txt" #"w_facebook_dir_info" # | "only_combined_txt"
    case = '_simple'

    if method == "w_facebook_dir_info":
        case = '_egdes_subnet_info'
        # 10 sub-networks
        for i in [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]:
            edges = []
            
            # Read the edges for the sub-network
            with open(f'facebook/{i}.edges') as f:
                for line in f:
                    u, v = map(int, line.strip().split())
                    edges.append((u, v))

            # Build sub-graph for the sub-network
            G_i = nx.Graph()
            G_i.add_edges_from(edges)
            graphs[i] = G_i

            # Detect communities using Label Propagation Algorithm
            communities = list(label_propagation_communities(G_i))
            
            # Plotting the subnetwork and its detected communities
            plt.figure(figsize=(10, 10))
            pos = nx.spring_layout(G_i, seed=42)
            
            # Assign colors to the communities
            cmap = plt.get_cmap('tab20')
            colors = [mcolors.to_rgba(cmap(i)) for i in range(len(communities))]
            
            for idx, community in enumerate(communities):
                nx.draw_networkx_nodes(G_i, pos, nodelist=list(community), node_size=50, node_color=[colors[idx]]*len(community))
            
            nx.draw_networkx_edges(G_i, pos, alpha=0.5)
            
            plt.title(f'Sub-network {i} Community Detection (Label Propagation)', fontsize=16)
            plt.axis('off')
            
            # Save each plot in the 'images' folder
            plt.savefig(f'./images/subnetwork_{i}_label_propagation.png')
            plt.show()

        # Combined the sub-networks into the full network
        for i in [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]:
            G.add_edges_from(graphs[i].edges())
    
    save_fig_name = './images/' + 'full_network_label_propagation'+case+'.png'

    # Full network community detection using Label Propagation

    communities_l = label_propagation_communities(G)
    communities = list(communities_l)
    aux.save_partition_with_pickle(communities, "label_propagation_partition.pkl")
    algorithm_time = time.time()
    print(communities)
    print(len(communities))

    pos = nx.spring_layout(G, seed=42)

    print("Number of communities:", len(communities))
    partition = {}
    for community_id, community in enumerate(communities):
        for node in community:
            partition[node] = community_id

    # Calculate modularity using the partition dictionary
    mod = community_louvain.modularity(partition, G)

    print(f'Modularity of the detected communities: {mod:.4f}')

    print("Time (s):", algorithm_time-init_time)

    # Plot the full network
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, seed=42)

    # Assign colors to the communities
    cmap =plt.get_cmap('tab20')
    # Assign colors to communities in the full network
    colors = [mcolors.to_rgba(cmap(i)) for i in range(len(communities))]
    for idx, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=list(community), node_size=10, node_color=[colors[idx]]*len(community)) # node_size=50

    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='lightgrey')
    # plt.title(f'Full Network Community Detection (Label Propagation)', fontsize=16)
    plt.axis('off')
    plt.savefig(save_fig_name, dpi=300)
    plt.show()

    loaded_partition = aux.load_partition_with_pickle("label_propagation_partition.pkl")

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

    # Load the ground truth clusters
    ground_truth = aux.load_ground_truth(cluster_files)

    # Evaluate clustering performance
    nmi, ari = aux.evaluate_clustering(partition, ground_truth, 'Louvain')
    print(f"Normalized Mutual Information (NMI): {nmi}")
    print(f"Adjusted Rand Index (ARI): {ari}")

    final_time = time.time()

    print("Time (s):", final_time-init_time)