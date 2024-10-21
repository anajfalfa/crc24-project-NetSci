### Auxiliar Functions

import networkx as nx
# pip install networkx
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# Create a graph from the input file
def create_graph_from_file(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            node1, node2 = map(int, line.split())
            G.add_edge(node1, node2)
    #print(G)
    return G

def save_partition_with_pickle(partition, filename):
    with open(filename, 'wb') as f:
        pickle.dump(partition, f)

def load_partition_with_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Visualize the communities
def visualize_communities(G, partition, method, figure_name="network.png"):
    # Create a list of community colors
    if method == "Louvain":
        community_colors = [partition[node] for node in G.nodes()]

    if method =="Infomap":
        # Criar um dicionário para mapear cada nó para sua respectiva comunidade
        node_to_community = {}
        for idx, community in enumerate(partition):
            for node in community:
                node_to_community[node] = idx  # Mapear cada nó para o índice da comunidade

        # Gerar uma lista de cores baseadas na comunidade de cada nó
        community_colors = [node_to_community[node] for node in G.nodes()]

    if method =="Giravan-Newman":
        community_colors = [partition[node] for node in G.nodes()]
    
    # Draw the graph
    pos = nx.spring_layout(G)  # For a better visualization layout
    #pos = nx.spring_layout(G, k=0.15, iterations=20) # Adjusted spring layout for spacing
    
    plt.figure(figsize=(15, 15))  # Larger figure size for clarity
    nx.draw(
        G, 
        pos, 
        node_color=community_colors, 
        cmap=plt.cm.rainbow,  # Fancy color map
        node_size=10,        # Smaller node size for clarity
        with_labels=False,    # No node labels for a cleaner look
        edge_color='lightgrey',    # Light gray edges for better visibility
        linewidths=0.001,       # Thinner edges for visual neatness
    )
    
    # Create a legend
    unique_communities = np.unique(partition.membership)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(len(unique_communities))]
    labels = [f"Community {community}" for community in unique_communities]
    
    # Add the legend to the plot
    plt.legend(handles, labels, title="Communities", loc='upper right', fontsize='small')

    # Step 4: Save the figure as "network.png" in high resolution
    plt.savefig("images/"+figure_name, dpi=300)  # Higher DPI for better resolution
    plt.show()

# Load ground truth clusters from files
def load_ground_truth(cluster_files):
    ground_truth = {}
    for cluster_id, file_path in enumerate(cluster_files):
        with open(file_path, 'r') as f:
            for line in f:
                nodes = map(int, line.split())
                for node in nodes:
                    ground_truth[node] = cluster_id
    return ground_truth


# Compare the partition and ground truth clusters
def evaluate_clustering(partition, ground_truth, method):
    ground_truth_labels = []

    if method == 'Infomap':
        # VertexClustering objects use membership to store cluster assignments
        partition_labels = partition.membership  # Get the cluster assignment for each node
            
        # Assume ground_truth is a dictionary or a similar structure
        for node in range(len(partition_labels)):  # Iterate over nodes by their index
            if node in ground_truth:
                ground_truth_labels.append(ground_truth[node])
            else:
                # Optional: handle nodes not in ground_truth (e.g., ignore or assign a default)
                ground_truth_labels.append(-1)  # -1 for nodes not in ground truth (optional)

    if method == 'Louvain':
        # Align the node labels from partition and ground_truth
        partition_labels = []
        
        for node in partition.keys():
            if node in ground_truth:  # Ignore nodes not in the ground truth
                partition_labels.append(partition[node])
                ground_truth_labels.append(ground_truth[node])

    if method == 'Girvan-Newman':
        # Handle the case where partition is a list of sets
        partition_labels = {}
        '''for community_id, community in enumerate(partition):
            for node in community:
                node_to_community[node] = community_id'''
        for node in partition:
            partition_labels[node] = partition[node]
        partition_labels = [partition_labels[node] for node in sorted(partition_labels.keys())] # Now it becomes a dictionary
        partition_labels=partition
        ground_truth_labels = ground_truth

        partition_labels = [partition.get(node, -1) for node in range(len(ground_truth_labels))]

    # print(len(partition_labels))
    # print(len(ground_truth_labels))

    # Use clustering evaluation metrics
    nmi = normalized_mutual_info_score(ground_truth_labels, partition_labels)
    ari = adjusted_rand_score(ground_truth_labels, partition_labels)
    

    return nmi, ari




