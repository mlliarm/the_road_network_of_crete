import networkx as nx
import matplotlib.pyplot as plt

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


def main():
    list_of_edges = parse_edges("crt_edges_wo_weights.txt")
    print(list_of_edges)

    G = create_network(list_of_edges)

    options = options = {
            'node_color': 'red',
            'node_size': 10,
            'width': 1,
            }
    #plt.subplot(221)
    #fig1 = plt.figure()
    #nx.draw_random(G, with_labels=True, **options)
    #fig1.savefig('crete_random.jpg')
    #plt.subplot(222)
    #fig2 = plt.figure()
    #nx.draw_circular(G, with_labels=True, **options)
    #fig2.savefig('crete_circular.jpg')
    #plt.subplot(223)
    #fig3 = plt.figure()
    #nx.draw_spectral(G,with_labels=True, **options)
    #fig3.savefig('crete_spectral.jpg')
    #plt.subplot(224)
    #nx.draw_shell(G, nlist=[range(7,10), range(68)], **options)
    #plt.show()

    DC = nx.algorithms.degree_centrality(G)
    print(DC)
    CC = nx.algorithms.closeness_centrality(G)
    print(CC)
    BC = nx.algorithms.betweenness_centrality(G)
    print(BC)
    EC = nx.algorithms.eigenvector_centrality(G)
    print(EC)
    HC = nx.algorithms.harmonic_centrality(G)
    print(HC)
    PC = nx.algorithms.pagerank(G)
    print(PC)
main()

