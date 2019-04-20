"""
The purpose of this script is to parse the graph data from a dataset of its edges,
analyse its statistical properties, calculate the most important centrality measures,
plot the network and the degree distributions. Also, comparisons are being made of given
input network with other well known complex networks as Erdos-Renyi and Small-World networks.

@author: Michail Liarmakopoulos

Example run:
	python3 model_cretan_road_network.py type_of_network

Prerequisities:
	networkx, matplotlib
"""

import networkx as nx
import matplotlib.pyplot as plt
import sys

def parse_edges(input_filename):
    with open(input_filename, 'r') as handle:
        list_of_edges = list()
        for line in handle:
            line = line.rstrip('\n')
            line = line.split('\t')
            line = tuple(line)
            line = (int(line[0]),int(line[1]))
            list_of_edges.append(line)
    return list_of_edges


def create_network(list_of_edges):
    G = nx.Graph()
    G.add_edges_from(list_of_edges)
    return G

def draw_network(G, type_of_network=None):
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
    plt.savefig("network_" + str(type_of_network) + ".png")

def calculation_of_centrality_measures(G):
	DC = nx.algorithms.degree_centrality(G)
    CC = nx.algorithms.closeness_centrality(G)
    BC = nx.algorithms.betweenness_centrality(G)
    EC = nx.algorithms.eigenvector_centrality(G)
    HC = nx.algorithms.harmonic_centrality(G)
    PC = nx.algorithms.pagerank(G)
    return (DC, CC, BC, EC, HC, PC)

def main():
    try:
        type_of_network = sys.argv[1]
    except IndexError:
        print("Please insert type of network from the below:")
        print("planar")
        print("circular")
        print("random")
        print("spectral")
        print("kamada_kawai")
        print("spring")
        print("shell")
        print("Type: python3 model_cretan_road_network.py type_of_network")
        return [0, "IndexError"]

    list_of_edges = parse_edges("data/crt_edges_wo_weights.txt")

    # Creation of the network
    G = create_network(list_of_edges)

    # Calculation of centrality measures
    DC, CC, BC, EC, HC, PC = calculation_of_centrality_measures(G)

    # Clustering
    clustering = nx.algorithms.clustering(G)

    # Printing info of graph
    print(nx.info(G))

    # Draw network
    draw_network(G, type_of_network)

    N, K = G.order(), G.size()
    avg_deg = float(K)/N

    print("Nodes:",N)
    print("Edges:",K)
    print("Average degree:",avg_deg)

if __name__ == "__main__":
    main()

