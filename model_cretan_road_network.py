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
from operator import itemgetter
import sys
from typing import List, Tuple, Dict

def parse_edges(input_filename: str) -> List[Tuple[str, str]]:
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


def parse_node_names(input_filename: str) -> Dict[str, str]:
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


def create_network_from_edges(list_of_edges: List[Tuple[str, str]]) -> nx.classes.graph.Graph:
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


def draw_network(G: nx.classes.graph.Graph, output_name: str, type_of_network: str=None) -> None:
    """
    Creates a drawing of the network, according to the selected type of network.

    Args:
        G (graph): the input graph
        output_name (string): the output name
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
    plt.savefig("images/" + output_name + "network_" + str(type_of_network) + ".png")
    plt.close()


def draw_ego_hub(G: nx.classes.graph.Graph, output_name: str) -> None:
    # From https://networkx.github.io/documentation/latest/auto_examples/drawing/plot_ego_graph.html?highlight=hub
    node_and_degree = G.degree()
    (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
    # Create ego graph of main hub
    hub_ego = nx.ego_graph(G, largest_hub)
    # Draw graph
    pos = nx.spring_layout(hub_ego)
    nx.draw(hub_ego, pos, node_color='b', node_size=300, with_labels=True)
    # Draw ego as large and red
    nx.draw_networkx_nodes(hub_ego, pos, nodelist=[largest_hub], node_size=300, node_color='r')
    plt.savefig("images/" + output_name + "ego_hub.png")
    plt.close()


def calculation_of_centrality_measures(G: nx.classes.graph.Graph) -> Tuple[float, float, float, float, float, float]:
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


def print_network_options() -> None:
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


def degrees_per_node(G: nx.classes.graph.Graph) -> List[Tuple[str, float]]:
    """
    Calculates the degrees per each node.

    Args:
        G (graph): input graph G

    Returns:
        nx.degree(G) (list): a list with tuples where each tuple has as first value the code name of the network
                             and as second the degree number
    """
    return nx.degree(G)


def calculate_average_degree(G: nx.classes.graph.Graph) -> float:
    N, K = G.order(), G.size()
    avg_deg = float(K)/N
    print("Nodes:",N)
    print("Edges:",K)
    print("Average degree:",avg_deg)
    return avg_deg


def draw_adjacency_matrix(G: nx.classes.graph.Graph, output_name: str, node_order=None, partitions=[], colors=[]) -> None:
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
    print('partitions:', partitions)
    print('node_order:', node_order)
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
    plt.savefig("images/" + output_name + "_adjacency_matrix.png")
    plt.close()


def create_histogram(G_histogram: List[int], output_name: str) -> None:
    """
    Inspired by: https://www3.nd.edu/~kogge/courses/cse60742-Fall2018/Public/StudentWork/Paradigms/NetworkX-Sikdar.pdf
    """
    xs = range(len(G_histogram))
    plt.scatter(xs, G_histogram)
    plt.xlabel('degree')
    plt.ylabel('counts')
    plt.savefig("images/" + output_name + "histogram.png")
    plt.close()


def plot_degree_distribution(G: nx.classes.graph.Graph, output_name: str) -> None:
    """
    Taken from: http://snap.stanford.edu/class/cs224w-2012/nx_tutorial.pdf
    """
    degs: Dict[int, int] = {}
    for n in G.nodes():
        deg = G.degree(n)
        if deg not in degs:
            degs[deg] = 0
        degs[deg] += 1
    items = sorted(degs.items())
    fig = plt.figure()
    ax = fig.add_subplot (111)
    ax.plot([k for (k,v) in items], [v for (k,v) in  items ],'bo--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.title(output_name + "Degree  Distribution")
    fig.savefig("images/" + output_name + "degree_distribution.png")
    plt.close()


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
    #draw_network(G, "crete", type_of_network)

    # Average degree
    avg_deg = calculate_average_degree(G)

    # Calculate the density of the graph
    density = nx.density(G)
    print("Density:",density)

    # Degree histogram
    dh = nx.degree_histogram(G)
    #print("Degree histogram:",dh)
    print('dh type:', type(dh))
    print('dh:',dh)
    create_histogram(dh, "crete_")

    # Degrees per node
    degrees = degrees_per_node(G)
    #print("Degrees per node:",degrees)

    # Adjacency matrix
    #adjacency_mat = nx.adjacency_matrix(G).todense()
    draw_adjacency_matrix(G, "Crete")

    # Degree distribution
    plot_degree_distribution(G, "Crete")

    # Diameter
    diameter = nx.algorithms.distance_measures.diameter(G)
    print("Crete diameter: ",diameter)

    # Find the biggest hub
    draw_ego_hub(G, "crete_")

    # Creation of some random graphs
    # Erdos Renyi
    erdos_renyi_G1 = nx.erdos_renyi_graph(68,0.1)
    draw_network(erdos_renyi_G1, "erdos_renyi_0.1_","circular")
    avg_deg_ER = calculate_average_degree(erdos_renyi_G1)
    dh_ER = nx.degree_histogram(erdos_renyi_G1)
    create_histogram(dh_ER, "erdos_renyi_0.1_")
    #print(dh_ER)
    degrees_ER = degrees_per_node(erdos_renyi_G1)
    #print(degrees_ER)
    plot_degree_distribution(erdos_renyi_G1, "Erdos-Renyi_0.1")
    draw_adjacency_matrix(erdos_renyi_G1, "Erdos-Renyi_0.1")
    # Diameter
    diameter_ER1 = nx.algorithms.distance_measures.diameter(erdos_renyi_G1)
    print("Erdos-Renyi_0.1 diameter: ",diameter_ER1)

    erdos_renyi_G2 = nx.erdos_renyi_graph(68,0.5)
    draw_network(erdos_renyi_G2, "erdos_renyi_0.5_","circular")
    avg_deg_ER2 = calculate_average_degree(erdos_renyi_G2)
    dh_ER2 = nx.degree_histogram(erdos_renyi_G2)
    create_histogram(dh_ER2, "erdos_renyi_0.5_")
    #print(dh_ER2)
    degrees_ER2 = degrees_per_node(erdos_renyi_G2)
    #print(degrees_ER2)
    plot_degree_distribution(erdos_renyi_G2, "Erdos-Renyi_0.5")
    draw_adjacency_matrix(erdos_renyi_G2, "Erdos-Renyi_0.5")
    # Diameter
    diameter_ER2 = nx.algorithms.distance_measures.diameter(erdos_renyi_G2)
    print("Erdos-Renyi_0.5 diameter: ",diameter_ER2)

    # # Scale free directed graph
    # scale_free_G = nx.scale_free_graph(68)
    # draw_network(scale_free_G, "scale_free_", "kamada_kawai")
    # dh_scalefree = nx.degree_histogram(scale_free_G)
    # create_histogram(dh_scalefree, "scale free_")
    # #print(dh_scalefree)
    # degrees_scalefree = degrees_per_node(scale_free_G)
    # #print(degrees_scalefree)
    # plot_degree_distribution(scale_free_G, "scale_free_")
    # draw_adjacency_matrix(scale_free_G, "scale_free_")
    # Diameter
    # diameter_scalefree = nx.algorithms.distance_measures.diameter(scale_free_G)
    # print("Scalefree network diameter: ",diameter_scalefree)

if __name__ == "__main__":
    main()

