import networkx as nx
import numpy as np

def candidate_mappings(G_dictionary_by_vertex, P_degrees):
    candidate_mappings = np.zeros((len(P_degrees), len(G_dictionary_by_vertex)))

    p_node_to_index = {p: i for i, p in enumerate(P_degrees.keys())}
    g_node_to_index = {g: i for i, g in enumerate(G_dictionary_by_vertex.keys())}
    for p, p_degree in P_degrees.items():
        for g, g_degree in G_dictionary_by_vertex.items():
            if p_degree <= g_degree:
                candidate_mappings[p_node_to_index.get(p)][g_node_to_index.get(g)] = 1
    return candidate_mappings


def ullman(G, P):
    P_degrees = dict(P.degree())
    G_degrees = dict(G.degree()) # this is G_dictionary_by_vertex
    G_dictionary_by_degree = {}


    max_degree = 0
    for g, g_deg in G_degrees.items():
        max_degree = max(max_degree, g_deg)
        G_dictionary_by_degree[g_deg] = G_dictionary_by_degree.get(g_deg, []) + [int(g)]

    for i in range(max_degree - 1, -1, -1):
        G_dictionary_by_degree[i] = G_dictionary_by_degree.get(i, []) + G_dictionary_by_degree.get(i+1, [])
    


    adj_list_G = {node: list(neighbors.keys()) for node, neighbors in G._adj.items()}
    adj_list_P = {node: list(neighbors.keys()) for node, neighbors in P._adj.items()}



    print("G_dictionary_by_vertex", G_degrees)
    print("G_dictionary_by_degree", G_dictionary_by_degree)

    print("Candidate mappings\n", candidate_mappings(G_degrees, P_degrees))


