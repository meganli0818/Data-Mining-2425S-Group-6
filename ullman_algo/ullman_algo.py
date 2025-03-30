import networkx as nx
import numpy as np

def ullman(G, P):
    P_degrees = dict(P.degree())
    G_degrees = dict(G.degree())
    G_dictionary_by_degree = {}


    max_degree = 0
    for g, g_deg in G_degrees.items():
        max_degree = max(max_degree, g_deg)
        G_dictionary_by_degree[g_deg] = G_dictionary_by_degree.get(g_deg, []) + [int(g)]

    for i in range(max_degree - 1, -1, -1):
        G_dictionary_by_degree[i] = G_dictionary_by_degree.get(i, []) + G_dictionary_by_degree.get(i+1, [])
    
    adj_list_G = {node: list(neighbors.keys()) for node, neighbors in G._adj.items()}
    adj_list_P = {node: list(neighbors.keys()) for node, neighbors in P._adj.items()}


    print("G_dictionary_by_degree", G_dictionary_by_degree)
