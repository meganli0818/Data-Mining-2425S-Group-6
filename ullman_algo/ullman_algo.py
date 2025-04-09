import networkx as nx
import numpy as np

class UllmanAlgorithm:
    def __init__(self, G, P):
        self.P_dictionary_by_vertex = dict(P.degree())
        self.G_dictionary_by_vertex = dict(G.degree())
        G_dictionary_by_degree = {}
        max_degree = 0
        for g, g_deg in self.G_dictionary_by_vertex.items():
            max_degree = max(max_degree, g_deg)
            G_dictionary_by_degree[g_deg] = G_dictionary_by_degree.get(g_deg, []) + [int(g)]

        for i in range(max_degree - 1, -1, -1):
            G_dictionary_by_degree[i] = G_dictionary_by_degree.get(i, []) + G_dictionary_by_degree.get(i+1, [])
        
        self.G_dictionary_by_degree = G_dictionary_by_degree
        self.adj_list_G = {node: list(neighbors.keys()) for node, neighbors in G._adj.items()}
        self.adj_list_P = {node: list(neighbors.keys()) for node, neighbors in P._adj.items()}


    def candidate_mappings(self):
        candidate_mappings = np.zeros((len(self.P_dictionary_by_vertex), len(self.G_dictionary_by_vertex)), dtype=bool) # store a boolean matrix to save space 

        p_node_to_index = {p: i for i, p in enumerate(self.P_dictionary_by_vertex.keys())}
        g_node_to_index = {g: i for i, g in enumerate(self.G_dictionary_by_vertex.keys())}
        for p, p_degree in self.P_dictionary_by_vertex.items():
            for g, g_degree in self.G_dictionary_by_vertex.items():
                if p_degree <= g_degree:
                    candidate_mappings[p_node_to_index.get(p)][g_node_to_index.get(g)] = 1
        return candidate_mappings



    def ullman(self):
        first_vertex = next(iter(self.P.nodes))
        return self.recursive_ullman(first_vertex, self.candidate_mappings(self.G_dictionary_by_vertex, self.P_dictionary_by_vertex), set())


    # x is the vertex we are currently matching
    def recursive_ullman(self, x, candidate_mapping_matrix, visited):
        x_neighbors = set(self.adj_list_P[x])
        unvisited_neighbors = x_neighbors - visited
        if not unvisited_neighbors:
            return True
        
        possible_matches = candidate_mapping_matrix[x]
        visited.add(x)

        for a in possible_matches:
            a_neighbors = set(self.adj_list_G[a])
            if all(any(candidate_mapping_matrix[unvisited_neighbor][a_neighbor] for a_neighbor in a_neighbors) for unvisited_neighbor in unvisited_neighbors):
                candidate_copy = np.copy(candidate_mapping_matrix)
                candidate_copy[:, a] = False
                next_x = next(iter(unvisited_neighbors))
                if self.recursive_ullman(next_x, candidate_copy, visited):
                    return True
        visited.remove(x)
        return False
        
