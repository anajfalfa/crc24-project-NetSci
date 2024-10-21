# Infomap algorithm

import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import igraph as ig
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
    
    method = "only_combined_txt" #"w_facebook_dir_info"  # | "only_combined_txt"
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

            # Convert NetworkX graph to igraph graph
            G_i_ig = ig.Graph.TupleList(G_i.edges(), directed=False)

            # Detect communities using Infomap Algorithm
            partition = G_i_ig.community_infomap()

            # Plotting the subnetwork and its detected communities
            plt.figure(figsize=(10, 10))
            pos = nx.spring_layout(G_i, seed=42)

            # Get the community membership list from the partition
            membership = partition.membership

            # Assign colors to the communities
            cmap = plt.get_cmap('tab20')
            unique_communities = set(membership)
            colors = {comm: mcolors.to_rgba(cmap(i)) for i, comm in enumerate(unique_communities)}

            # Draw nodes with community-specific colors
            for idx, node in enumerate(G_i.nodes()):
                community_id = membership[idx]
                nx.draw_networkx_nodes(G_i, pos, nodelist=[node], node_size=50, node_color=[colors[community_id]])

            nx.draw_networkx_edges(G_i, pos, alpha=0.5)

            plt.title(f'Sub-network {i} Community Detection (Infomap)', fontsize=16)
            plt.axis('off')

            # plt.savefig(f'./images/subnetwork_{i}_infomap.png')
            plt.show()

        # Combine the sub-networks into the full network
        for i in [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]:
            G.add_edges_from(graphs[i].edges())

    save_fig_name = './images/full_network_infomap' + case + '.png'

    # Convert the full NetworkX graph to igraph
    G_ig = ig.Graph.TupleList(G.edges(), directed=False)

    # Full network community detection using Infomap
    partition = G_ig.community_infomap()
    aux.save_partition_with_pickle(partition, "infomap_partition.pkl")
    algorithm_time = time.time()

    print("Number of communities:", len(partition))

    print("Time (s):", algorithm_time-init_time)

    partition_dict = {v.index: community_id for v, community_id in zip(G_ig.vs, partition.membership)}

    # Calculate modularity using the partition dictionary
    mod = community_louvain.modularity(partition_dict, G)

    print(f'Modularity of the detected communities: {mod:.4f}')

    # Get the community membership list
    membership = partition.membership

    # Plot the full network
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, seed=42)

    # Assign colors to the communities
    cmap =plt.get_cmap('tab20') #plt.cm.rainbow  #plt.get_cmap('tab20')
    unique_communities = set(membership)
    colors = {comm: mcolors.to_rgba(cmap(i)) for i, comm in enumerate(unique_communities)}

    # Draw nodes with community-specific colors
    for idx, node in enumerate(G.nodes()):
        community_id = membership[idx]
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=10, node_color=[colors[community_id]])

    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='lightgrey')
    # plt.title(f'Full Network Community Detection (Infomap)', fontsize=16)
    plt.axis('off')
    plt.savefig(save_fig_name, dpi=300)
    plt.show()

    loaded_partition = aux.load_partition_with_pickle("infomap_partition.pkl")


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
    nmi, ari = aux.evaluate_clustering(loaded_partition, ground_truth, 'Infomap')
    print(f"Normalized Mutual Information (NMI): {nmi}")
    print(f"Adjusted Rand Index (ARI): {ari}")

    final_time = time.time()
    print("Time (s):", final_time - init_time)