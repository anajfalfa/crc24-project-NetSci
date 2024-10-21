# Ground Truth

import os
from community import community_louvain
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.quality import modularity
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import igraph as ig
import time
import aux_functions as aux

# Main function
if __name__ == "__main__":
    init_time = time.time()
    np.random.seed(42)  # Set seed for NumPy

    output_dir = './images'

    graphs = {}
    with open('./facebook_combined.txt') as f:
        G = nx.Graph()
        for line in f:
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)

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
    ground_truth = aux.load_ground_truth(cluster_files)
    # print(ground_truth)
    # print("Ground Truth Keys:", list(ground_truth.keys()))

    for node in G.nodes():
        if node not in ground_truth:
            ground_truth[node] = -1  # Assign -1 or any other value to represent unassigned nodes

    unique_labels = list(set(ground_truth.values()))
    print(unique_labels)
    print(len(unique_labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    ground_truth_normalized = {node: label_mapping[community] for node, community in ground_truth.items()}

    mod = community_louvain.modularity(ground_truth, G) 

    print(f'Modularity of the detected communities: {mod:.4f}')
    # Modularity of the detected communities: 0.6696
    algorithm_time = time.time()

    print("Time (s):", algorithm_time-init_time)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)

    unique_communities = set(ground_truth.values())
    cmap =plt.cm.rainbow  #plt.get_cmap('tab20')
    cmap =plt.get_cmap('tab20')

    colors = {comm: mcolors.to_rgba(cmap(i)) for i, comm in enumerate(unique_communities)}

    # Draw nodes with community-specific colors
    for node in G.nodes():
        community_id = ground_truth[node]
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=10, node_color=[colors[community_id]])

    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='lightgrey')
    #plt.title('Community Detection (Ground Truth)', fontsize=16)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'ground_truth_community_detection.png'))
    plt.show()

    # Evaluate clustering performance
    nmi, ari = aux.evaluate_clustering(ground_truth, ground_truth, 'Louvain')
    
    print(f"Normalized Mutual Information (NMI): {nmi}")
    print(f"Adjusted Rand Index (ARI): {ari}")

    final_time = time.time()
    print("Time (s):", final_time-init_time)