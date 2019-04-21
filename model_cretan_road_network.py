"""
The purpose of this script is to parse the graph data from a dataset of its edges,
analyse its statistical properties, calculate the most important centrality measures,
plot the network and the degree distributions. Also, comparisons are being made of given
input network with other well known complex networks as Erdos-Renyi and Small-World networks.

@author: Michail Liarmakopoulos (unless stated otherwise)

Example run:
    python3 model_cretan_road_network.py type_of_network

Prerequisities:
    networkx, matplotlib,numpy, scipy
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pyplot, patches
import scipy as sp
import numpy as np
import sys

def parse_edges(input_filename):
    """
    Parses the edges into a pair and returns a list of pairs.

    Args:
        input_filename (string): a tab delimited input file with the pairs of the nodes that create an edge

    Returns:
        list_of_edges (list): list with pairs of coded node names
    """
    with open(input_filename, 'r') as handle:
        list_of_edges = list()
        for line in handle:
            line = line.rstrip('\n')
            line = line.split('\t')
            line = tuple(line)            
            list_of_edges.append(line)
    return list_of_edges

def parse_node_names(input_filename):
    """
    Parses node names and returns a dictionary with the code name as a key and its real name as a value.

    Args:
        input_filename (string): a tab delimited input file with the pairs of the coded name and real name

    Returns:
        dict_of_nodes (dictionary): a dictionary with keys the coded name and values the corresponding real name
    """
    with open(input_filename, 'r') as handle:
        dict_of_nodes = dict()
        for line in handle: 
            line = line.rstrip('\n')
            line = line.split('\t')
            dict_of_nodes[line[0]] = line[1]
    return dict_of_nodes

def create_network_from_edges(list_of_edges):
    """
    Creates a network given a list of pairs with the edges.

    Args:
        list_of_edges (list): the list with the pairs of edges

    Returns:
        G (graph): the resulting graph/network
    """
    G = nx.Graph()
    G.add_edges_from(list_of_edges)
    return G

def draw_network(G, type_of_network=None):
    """
    Creates a drawing of the network, according to the selected type of network.

    Args:
        G (graph): the input graph
        type_of_network (string): the type of network

    Returns:
        None. Just prints the image to a file into the folder data/
    """
    if type_of_network == "planar":
        nx.draw_planar(G, with_labels = True)
    elif type_of_network == "circular":
        nx.draw_circular(G, with_labels = True)
    elif type_of_network == "random":
        nx.draw_random(G, with_labels = True)
    elif type_of_network == "spectral":
        nx.draw_random(G, with_labels = True)
    elif type_of_network == "kamada_kawai":
        nx.draw_kamada_kawai(G, with_labels = True)
    elif type_of_network == "spring":
        nx.draw_spring(G, with_labels = True)
    elif type_of_network == "shell":
        nx.draw_shell(G, with_labels = True)
    else:
        nx.draw(G, with_labels = True)
    plt.savefig("images/" + "network_" + str(type_of_network) + ".png")

def calculation_of_centrality_measures(G):
    """
    Calculates the centrality measures for a given graph G.

    Args:
        G (graph): input graph

    Returns:
        (DC, CC, BC, EC, HC, PC) (tuple): a tuple with the resulting centrality measures
    """
    DC = nx.algorithms.degree_centrality(G)
    CC = nx.algorithms.closeness_centrality(G)
    BC = nx.algorithms.betweenness_centrality(G)
    EC = nx.algorithms.eigenvector_centrality(G)
    HC = nx.algorithms.harmonic_centrality(G)
    PC = nx.algorithms.pagerank(G)
    return (DC, CC, BC, EC, HC, PC)

def print_network_options():
    """
    Prints the options for the drawing function when run from console.
    """
    print("Please insert type of network from the below:")
    print("planar")
    print("circular")
    print("random")
    print("spectral")
    print("kamada_kawai")
    print("spring")
    print("shell")
    print("Type: python3 model_cretan_road_network.py type_of_network")

def degrees_per_node(G):
    """
    Calculates the degrees per each node.

    Args:
        G (graph): input graph G

    Returns:
        nx.degree(G) (list): a list with tuples where each tuple has as first value the code name of the network
                             and as second the degree number
    """
    return nx.degree(G)


def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):
    """
    From : http://sociograph.blogspot.com/2012/11/visualizing-adjacency-matrices-in-python.html
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")
    
    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)
    plt.savefig("images/adjacency_matrix.png")

def main():
    try:
        type_of_network = sys.argv[1]
    except IndexError:
        print_network_options()        
        return [0, "IndexError"]

    # Parse undirected edges to list of tuples
    list_of_edges = parse_edges("data/crt_edges_wo_weights.txt")

    # Parse names
    names_dict = parse_node_names("data/crt_vertices_names2.txt")
    print(names_dict)

    # Creation of the network
    G = create_network_from_edges(list_of_edges)

    # Calculation of centrality measures
    DC, CC, BC, EC, HC, PC = calculation_of_centrality_measures(G)

    # Clustering
    clustering = nx.algorithms.clustering(G)

    # Printing info of graph
    print(nx.info(G))

    # Draw network
    #draw_network(G, type_of_network)

    N, K = G.order(), G.size()
    avg_deg = float(K)/N

    print("Nodes:",N)
    print("Edges:",K)
    print("Average degree:",avg_deg)

    # Calculate the density of the graph
    density = nx.density(G)
    print("Density:",density)

    # Degree histogram
    dh = nx.degree_histogram(G)
    print("Degree histogram:",dh)

    # Degrees per node
    degrees = degrees_per_node(G)
    print("Degrees per node:",degrees)

    # Adjacency matrix
    #adjacency_mat = nx.adjacency_matrix(G).todense()
    draw_adjacency_matrix(G)

if __name__ == "__main__":
    main()

