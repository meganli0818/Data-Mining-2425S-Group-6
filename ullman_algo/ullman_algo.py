import networkx as nx
import numpy as np

class UllmanAlgorithm:
    def __init__(self, G, P):
        if len(P.edges()) > len(G.edges()):
            raise ValueError("P cannot be larger than G")
        if len(P.nodes()) > len(G.nodes()):
            raise ValueError("P cannot be larger than G")
        self.P_dictionary_by_vertex = dict(P.degree())
        self.G_dictionary_by_vertex = dict(G.degree())
        self.G_vertices = set(G.nodes())
        self.P_vertices = set(P.nodes())
        G_dictionary_by_degree = {}
        max_degree = 0
        for g, g_deg in self.G_dictionary_by_vertex.items():
            max_degree = max(max_degree, g_deg)
            G_dictionary_by_degree[g_deg] = G_dictionary_by_degree.get(g_deg, []) + [int(g)]

        for i in range(max_degree - 1, -1, -1):
            G_dictionary_by_degree[i] = G_dictionary_by_degree.get(i, []) + G_dictionary_by_degree.get(i+1, [])
        
        self.G_dictionary_by_degree = G_dictionary_by_degree

        # Create mappings between original node IDs and matrix indices
        self.p_node_to_index = {p: i for i, p in enumerate(P.nodes())}
        self.g_node_to_index = {g: i for i, g in enumerate(G.nodes())}
        self.p_index_to_node = {i: p for p, i in self.p_node_to_index.items()}
        self.g_index_to_node = {i: g for g, i in self.g_node_to_index.items()}
        
        # Create adjacency lists using original node IDs
        self.adj_list_G = {node: list(neighbors) for node, neighbors in G.adjacency()}
        self.adj_list_P = {node: list(neighbors) for node, neighbors in P.adjacency()}
        
        self.visited = {}

    def candidate_mappings(self):
        candidate_mappings = np.zeros((len(self.P_dictionary_by_vertex), len(self.G_dictionary_by_vertex)), dtype=bool) # store a boolean matrix to save space 

        p_node_to_index = {p: i for i, p in enumerate(self.P_dictionary_by_vertex.keys())}
        g_node_to_index = {g: i for i, g in enumerate(self.G_dictionary_by_vertex.keys())}
        for p, p_degree in self.P_dictionary_by_vertex.items():
            for g, g_degree in self.G_dictionary_by_vertex.items():
                if p_degree <= g_degree:
                    candidate_mappings[p_node_to_index.get(p)][g_node_to_index.get(g)] = 1
        return candidate_mappings

    def get_unmapped_vertices(self):
        return set(self.G_vertices) - set(self.visited.values())

    def ullman(self):
        first_vertex = next(iter(self.adj_list_P.keys()))
        return self.recursive_ullman(first_vertex, self.candidate_mappings(), self.visited)


    # x is the vertex we are currently matching
    def recursive_ullman(self, x, candidate_mapping_matrix, visited):
        # Convert node ID to matrix index when accessing the matrix
        x_idx = self.p_node_to_index[x]
        
        # Base case: if all nodes are visited
        if len(visited) == len(self.adj_list_P):
            return True
            
        visited_nodes = set(visited.keys())
        x_neighbors = set(self.adj_list_P[x])
        unvisited_neighbors = x_neighbors - visited_nodes
        
        # Get possible matches using matrix index
        possible_matches = np.where(candidate_mapping_matrix[x_idx])[0]

        # Convert matrix indices back to node IDs when needed
        for a_idx in possible_matches:
            a = self.g_index_to_node[a_idx]
            a_neighbors = set(self.adj_list_G[a])
            
            # Check if neighbors can be matched
            if all(any(candidate_mapping_matrix[self.p_node_to_index[unvisited_neighbor]][self.g_node_to_index[a_neighbor]] for a_neighbor in a_neighbors) for unvisited_neighbor in unvisited_neighbors):
                candidate_copy = np.copy(candidate_mapping_matrix)
                candidate_copy[:, a_idx] = False
                
                # Apply constraints to neighbors
                for unvisited_neighbor in unvisited_neighbors:
                    unvisited_idx = self.p_node_to_index[unvisited_neighbor]
                    for i in range(len(self.g_node_to_index)):
                        if self.g_index_to_node[i] not in a_neighbors:
                            candidate_copy[unvisited_idx][i] = False
                
                # Map current vertex
                visited[x] = a
                
                # Choose next vertex to match
                if unvisited_neighbors:
                    next_x = next(iter(unvisited_neighbors))
                    if self.recursive_ullman(next_x, candidate_copy, visited):
                        return True
                else:
                    # No neighbors left, but still more vertices to match
                    leftoverVertices = set(self.adj_list_P.keys()) - set(visited.keys())
                    if not leftoverVertices:  # All vertices matched
                        return True
                    next_x = next(iter(leftoverVertices))
                    if self.recursive_ullman(next_x, candidate_copy, visited):
                        return True
                
                # Remove mapping if this branch fails
                del visited[x]
                
        return False

