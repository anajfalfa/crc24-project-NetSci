# Girvan-Newman Algorithm

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from community import community_louvain
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.quality import modularity
import time
import aux_functions as aux

init_time = time.time()
np.random.seed(42)  # Set seed for NumPy

# Read the file and build the graph
graphs = {}
G = nx.Graph()

with open('./facebook_combined.txt') as f:
    for line in f:
        u, v = map(int, line.strip().split())
        G.add_edge(u, v)

# Partition the whole network and calculate the modularity
pos = nx.spring_layout(G, seed=42)

comp = girvan_newman(G)
comp = sorted(next(comp))

aux.save_partition_with_pickle(comp, "partition_girvan_newman.pkl")
algorithm_time = time.time()
print("Number of communities:", len(comp))

print("Time (s):", algorithm_time-init_time)
partition = {}
for k, c in enumerate(comp):
    for node in c:
        partition[node] = k
modularity = community_louvain.modularity(partition, G)
print('Modularity of full network：', modularity)

case = '_simple'
save_fig_name = './images/' + 'full_network_girvan_newman'+case+'.png'

# Plot the complete network graph
plt.figure(figsize=(20, 20))
pos = nx.spring_layout(G, seed=42)
cmap =plt.get_cmap('tab20')
colors = [mcolors.to_rgba(cmap(partition[i])) for i in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=10, node_color=colors)
nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='lightgrey')
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
    
loaded_partition = aux.load_partition_with_pickle("partition_girvan_newman.pkl")
loaded_partition_dict = {}

for cluster_id, community in enumerate(loaded_partition):
    for node in community:
        loaded_partition_dict[node] = cluster_id

# Load the ground truth clusters
ground_truth = {}
for cluster_id, file_path in enumerate(cluster_files):
    with open(file_path, 'r') as f:
        for line in f:
            nodes = map(int, line.split())
            for node in nodes:
                ground_truth[node] = cluster_id
# ground_truth = aux.load_ground_truth(cluster_files)

ground_truth_labels = [ground_truth[node] for node in sorted(ground_truth.keys())]

# Evaluate clustering performance
nmi, ari = aux.evaluate_clustering(loaded_partition_dict, ground_truth_labels, 'Girvan-Newman')
print(f"Normalized Mutual Information (NMI): {nmi}")
print(f"Adjusted Rand Index (ARI): {ari}")
final_time = time.time()

print("Time (s):", final_time-init_time)