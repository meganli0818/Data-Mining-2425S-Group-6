import networkx as nx
import numpy as np

class UllmanAlgorithmEdge:
    """
    Implementation of Ullman's Subgraph Isomorphism Algorithm.
    
    This algorithm determines if a pattern graph P is a subgraph of a larger graph G.
    It uses a recursive backtracking approach with matrix-based constraint propagation.
    
    Attributes:
        P_dictionary_by_vertex: Dictionary mapping vertices in P to their degrees
        G_dictionary_by_vertex: Dictionary mapping vertices in G to their degrees
        G_vertices: Set of all vertices in G
        P_vertices: Set of all vertices in P
        G_dictionary_by_degree: Dictionary of vertices in G organized by degree
        p_node_to_index, g_node_to_index: Mappings from node IDs to matrix indices
        p_index_to_node, g_index_to_node: Mappings from matrix indices to node IDs
        adj_list_G, adj_list_P: Adjacency lists for graphs G and P
        visited: Dictionary tracking the current mapping from P to G
    """
    
    def __init__(self, G, P):
        """
        Initialize the Ullman algorithm with graphs G and P.
        
        Args:
            G: The larger graph (NetworkX graph object)
            P: The pattern graph to find within G (NetworkX graph object)
            
        Raises:
            ValueError: If P has more edges or vertices than G
        """
            
        # Store degree information for each vertex
        self.P_dictionary_by_vertex = dict(P.degree())
        self.G_dictionary_by_vertex = dict(G.degree())
        self.G_vertices = set(G.nodes())
        self.P_vertices = set(P.nodes())

        self.G_labels = nx.get_node_attributes(G, 'label')
        self.P_labels = nx.get_node_attributes(P, 'label')

        self.failure_count = 0
        
        # Group G's vertices by degree for efficient matching
        G_dictionary_by_degree = {}
        max_degree = 0
        for g, g_deg in self.G_dictionary_by_vertex.items():
            max_degree = max(max_degree, g_deg)
            G_dictionary_by_degree[g_deg] = G_dictionary_by_degree.get(g_deg, []) + [int(g)]

        # Create cumulative degree lists (vertices with degree >= i)
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

        self.P_edges = P.edges()
        self.G_edges = G.edges()

        self.P_node_to_edges = {node: set(edge for edge in self.P_edges if node in edge) for node in P.nodes()}
        
        self.visited_nodes = {}
        self.visited_edges = {}

    def candidate_mappings(self):
        """
        Generate initial candidate mapping matrix based on vertex degrees.
        
        A vertex p in P can be mapped to vertex g in G if degree(p) <= degree(g).
        
        Returns:
            numpy.ndarray: Boolean matrix where True indicates a potential mapping
        """
        # Create a boolean matrix to represent candidate mappings
        candidate_mappings = np.zeros((len(self.P_dictionary_by_vertex), len(self.G_dictionary_by_vertex)), dtype=bool)

        # Iterate through the degrees of P and G to find candidate mappings
        p_node_to_index = {p: i for i, p in enumerate(self.P_dictionary_by_vertex.keys())}
        g_node_to_index = {g: i for i, g in enumerate(self.G_dictionary_by_vertex.keys())}
        for p, p_degree in self.P_dictionary_by_vertex.items():
            for g, g_degree in self.G_dictionary_by_vertex.items():
                # vertex p in P can be mapped to vertex g in G if the degree of p <= degree of g and their labels match
                if p_degree <= g_degree and self.P_labels[p] == self.G_labels[g]:
                    candidate_mappings[p_node_to_index.get(p)][g_node_to_index.get(g)] = True
        return candidate_mappings
    
    def exact_candidate_mappings(self):
        """
        Generate initial candidate mapping matrix based on vertex degrees.
        
        A vertex p in P can be mapped to vertex g in G if degree(p) <= degree(g).
        
        Returns:
            numpy.ndarray: Boolean matrix where True indicates a potential mapping
        """
        # Create a boolean matrix to represent candidate mappings
        candidate_mappings = np.zeros((len(self.P_dictionary_by_vertex), len(self.G_dictionary_by_vertex)), dtype=bool)

        # Iterate through the degrees of P and G to find candidate mappings
        p_node_to_index = {p: i for i, p in enumerate(self.P_dictionary_by_vertex.keys())}
        g_node_to_index = {g: i for i, g in enumerate(self.G_dictionary_by_vertex.keys())}
        for p, p_degree in self.P_dictionary_by_vertex.items():
            for g, g_degree in self.G_dictionary_by_vertex.items():
                # vertex p in P can be mapped to vertex g in G if the degree of p <= degree of g
                if p_degree == g_degree and self.P_labels[p] == self.G_labels[g]:
                    candidate_mappings[p_node_to_index.get(p)][g_node_to_index.get(g)] = True
        return candidate_mappings


    def get_unmapped_vertices_in_G(self):
        """
        Get vertices in G that were not mapped to any vertex in P.
        
        Returns:
            set: Vertices in G not included in the mapping
        """
        return set(self.G_vertices) - set(self.visited_nodes.values())
    
    def get_unmapped_vertices_in_P(self):
        """
        Get vertices in G that were not mapped to any vertex in P.
        
        Returns:
            set: Vertices in G not included in the mapping
        """
        return set(self.P_vertices) - set(self.visited_nodes.keys())

    def get_unmapped_edges_in_G(self):
        """
        Get edges in G that were not mapped to any edge in P.
        
        Returns:
            set: Edges in G not included in the mapping
        """

        visited_edges_values = set(self.visited_edges.values())
        visited_edges_values = visited_edges_values | {(v,u) for (u,v) in visited_edges_values}
        return set(self.G_edges) - set(visited_edges_values)
    
    def get_unmapped_edges_in_P(self):
        """
        Get edges in P that were not mapped to any edge in G.
        
        Returns:
            set: Edges in P not included in the mapping
        """
        visited_edges_keys = set(self.visited_edges.keys())
        visited_edges_keys = visited_edges_keys | {(v,u) for (u,v) in visited_edges_keys}
        return set(self.P_edges) - set(visited_edges_keys)

    
    def get_mapping(self):
        """
        Get the current mapping of vertices from P to G.
        
        Returns:
            dict: Mapping of vertices in P to vertices in G
        """
        return self.visited_nodes

    def ullman(self, exact_match):
        """
        Execute Ullman's algorithm to find if P is a subgraph of G, or if they are isomorphic.

        Parameters:
            exact_match (bool): If True, find an exact match (same degree), otherwise find a subgraph match
        
        Returns:
            bool: True if P is a subgraph of G, False otherwise
        """
        if len(self.adj_list_P) == 0:
            return True
        first_edge = next(iter(sorted(self.P_edges)))
        if exact_match:
            candidate_mapping_matrix = self.exact_candidate_mappings()
        else:
            candidate_mapping_matrix = self.candidate_mappings()
        return self.recursive_ullman(first_edge, candidate_mapping_matrix, start=True)

    def recursive_ullman(self, e, candidate_mapping_matrix, start=False):
        """
        Recursive function to find a mapping from P to G.
        
        Args:
            x: Current vertex in P to match
            candidate_mapping_matrix: Matrix of possible mappings
            visited: Dictionary mapping vertices in P to vertices in G
            
        Returns:
            bool: True if a valid mapping was found, False otherwise
        """
        # Convert node ID to matrix index when accessing the matrix
        u, v = e
        u_idx = self.p_node_to_index[u]
        v_idx = self.p_node_to_index[v]

        # ADD A DATSTRUCTUE WHICH STORES WHAT EDGES TOUGH EACH VERTEX
        
        # Base case: if all nodes are visited
        if len(self.visited_edges.keys()) == len(self.P_edges):
            return True
            
        # Get the list of nodes we still need to map
        visited_edges_keys = set(self.visited_edges.keys())
        visited_edges_keys = visited_edges_keys | {(v,u) for (u,v) in visited_edges_keys}
        e_neighbors = self.P_node_to_edges[u] | self.P_node_to_edges[v]
        unvisited_neighboring_edges = e_neighbors - visited_edges_keys

        u_unvisited_neighbors = set(self.adj_list_P[u]) - set(self.visited_nodes.keys())
        v_unvisited_neighbors = set(self.adj_list_P[v]) - set(self.visited_nodes.keys())

        visited_nodes_keys = set(self.visited_nodes.keys())
        
        # if both vertices are already matched, check if they have an edge between the nodes they are mapped to
        if u in visited_nodes_keys and v in visited_nodes_keys:
            # Both vertices are already matched, check if they have an edge between them
            if (self.visited_nodes[u],self.visited_nodes[v]) in self.G_edges or (self.visited_nodes[v],self.visited_nodes[u]) in self.G_edges:
                self.visited_edges[e] = (self.visited_nodes[u], self.visited_nodes[v])
                if (len(unvisited_neighboring_edges)>=1):
                    # Choose the next edge to match
                    next_edge = next(iter(sorted(unvisited_neighboring_edges)))
                    if self.recursive_ullman(next_edge, candidate_mapping_matrix):
                        return True
                    else:
                        return False
                else:
                    next_edge = next(iter(sorted(self.P_edges - visited_edges_keys)))
                    if self.recursive_ullman(next_edge, candidate_mapping_matrix):
                        return True
            else:
                return False
        

        # if both vertices are not matched, try matching both
        if u not in visited_nodes_keys and v not in visited_nodes_keys:
            # Get possible matches using matrix index
            u_possible_matches = np.where(candidate_mapping_matrix[u_idx])[0]

            # Try each possible match for vertex u
            for a_idx in u_possible_matches:
                a = self.g_index_to_node[a_idx]
                a_neighbors = set(self.adj_list_G[a])
                
                # Check if all unvisited neighbors of u can be matched to neighbors of a
                if all(any(candidate_mapping_matrix[self.p_node_to_index[u_neighbor]][self.g_node_to_index[a_neighbor]] 
                            for a_neighbor in a_neighbors) 
                    for u_neighbor in u_unvisited_neighbors):
                    
                     # match v too
                    v_possible_matches = set(np.where(candidate_mapping_matrix[v_idx])[0])
                    v_possible_matches = v_possible_matches & {self.g_node_to_index[neighbor] for neighbor in a_neighbors}
                    
                    for b_idx in v_possible_matches:
                        b = self.g_index_to_node[b_idx]
                        b_neighbors = set(self.adj_list_G[b])     

                        # Check if all unvisited neighbors of v can be matched to neighbors of b
                        if all(any(candidate_mapping_matrix[self.p_node_to_index[v_neighbor]][self.g_node_to_index[b_neighbor]] 
                                for b_neighbor in b_neighbors) 
                            for v_neighbor in v_unvisited_neighbors):
                    
                            # Create a new constraint matrix
                            candidate_copy = np.copy(candidate_mapping_matrix)
                            candidate_copy[:, a_idx] = False  # Mark vertex a as used
                            candidate_copy[:, b_idx] = False  # Mark vertex b as used
                            
                            # Apply constraints: neighbors of u can only match to neighbors of a
                            for u_unvisited_neighbor in u_unvisited_neighbors-{v}:
                                unvisited_idx = self.p_node_to_index[u_unvisited_neighbor]
                                for i in range(len(self.g_node_to_index)):
                                    if self.g_index_to_node[i] not in a_neighbors:
                                        candidate_copy[unvisited_idx][i] = False

                            # Apply constraints: neighbors of v can only match to neighbors of b
                            for v_unvisited_neighbor in v_unvisited_neighbors-{u}:
                                unvisited_idx = self.p_node_to_index[v_unvisited_neighbor]
                                for i in range(len(self.g_node_to_index)):
                                    if self.g_index_to_node[i] not in b_neighbors:
                                        candidate_copy[unvisited_idx][i] = False
                            
                            # Add current mapping
                            self.visited_nodes[u] = a
                            self.visited_nodes[v] = b
                            self.visited_edges[e] = (a,b)
                            
                            # Choose next vertex to match (prioritize neighbors)
                            if unvisited_neighboring_edges:
                                next_edge = next(iter(sorted(unvisited_neighboring_edges)))
                                if self.recursive_ullman(next_edge, candidate_copy):
                                    return True
                            else:
                                next_edge = next(iter(sorted(self.P_edges - visited_edges_keys)))
                                if self.recursive_ullman(next_edge, candidate_copy):
                                    return True
                    
                            # Remove mapping if this branch fails (backtrack)
                            del self.visited_edges[e]
                            del self.visited_nodes[v]
                            del self.visited_nodes[u]
                    

                    if start:
                        self.failure_count += 1
                        if self.failure_count%100 == 0:
                            print(f"\rUllman failures: {self.failure_count}        ", end="")
            
        elif u not in visited_nodes_keys:
            # Get possible matches using matrix index
            u_possible_matches = np.where(candidate_mapping_matrix[u_idx])[0]

            # Try each possible match for vertex u
            for a_idx in u_possible_matches:
                a = self.g_index_to_node[a_idx]
                a_neighbors = set(self.adj_list_G[a])
                
                # Check if all unvisited neighbors of u can be matched to neighbors of a
                if all(any(candidate_mapping_matrix[self.p_node_to_index[u_neighbor]][self.g_node_to_index[a_neighbor]] 
                            for a_neighbor in a_neighbors) 
                    for u_neighbor in u_unvisited_neighbors):
                
                        # Create a new constraint matrix
                        candidate_copy = np.copy(candidate_mapping_matrix)
                        candidate_copy[:, a_idx] = False  # Mark vertex a as used
                        
                        # Apply constraints: neighbors of u can only match to neighbors of a
                        for u_unvisited_neighbor in u_unvisited_neighbors:
                            unvisited_idx = self.p_node_to_index[u_unvisited_neighbor]
                            for i in range(len(self.g_node_to_index)):
                                if self.g_index_to_node[i] not in a_neighbors:
                                    candidate_copy[unvisited_idx][i] = False
                        
                        # Add current mapping
                        self.visited_nodes[u] = a
                        self.visited_edges[e] = (a,self.visited_nodes[v])
                        
                        # Choose next vertex to match (prioritize neighbors)
                        if unvisited_neighboring_edges:
                            next_edge = next(iter(sorted(unvisited_neighboring_edges)))
                            if self.recursive_ullman(next_edge, candidate_copy):
                                return True
                        else:
                            next_edge = next(iter(sorted(self.P_edges - visited_edges_keys)))
                            if self.recursive_ullman(next_edge, candidate_copy):
                                return True
                
                        # Remove mapping if this branch fails (backtrack)
                        del self.visited_edges[e]
                        del self.visited_nodes[u]

                if start:
                    self.failure_count += 1
                    if self.failure_count%100 == 0:
                        print(f"\rUllman failures: {self.failure_count}        ", end="")
        elif v not in visited_nodes_keys:
            # Get possible matches using matrix index
            v_possible_matches = np.where(candidate_mapping_matrix[v_idx])[0]

            # Try each possible match for vertex v
            for b_idx in v_possible_matches:
                b = self.g_index_to_node[b_idx]
                b_neighbors = set(self.adj_list_G[b])
                
                # Check if all unvisited neighbors of v can be matched to neighbors of b
                if all(any(candidate_mapping_matrix[self.p_node_to_index[v_neighbor]][self.g_node_to_index[b_neighbor]] 
                            for b_neighbor in b_neighbors) 
                    for v_neighbor in v_unvisited_neighbors):
                
                        # Create a new constraint matrix
                        candidate_copy = np.copy(candidate_mapping_matrix)
                        candidate_copy[:, b_idx] = False  # Mark vertex b as used
                        
                        # Apply constraints: neighbors of v can only match to neighbors of b
                        for v_unvisited_neighbor in v_unvisited_neighbors:
                            unvisited_idx = self.p_node_to_index[v_unvisited_neighbor]
                            for i in range(len(self.g_node_to_index)):
                                if self.g_index_to_node[i] not in b_neighbors:
                                    candidate_copy[unvisited_idx][i] = False
                        
                        # Add current mapping
                        self.visited_nodes[v] = b
                        self.visited_edges[e] = (self.visited_nodes[u], b)
                        
                        # Choose next vertex to match (prioritize neighbors)
                        if unvisited_neighboring_edges:
                            next_edge = next(iter(sorted(unvisited_neighboring_edges)))
                            if self.recursive_ullman(next_edge, candidate_copy):
                                return True
                        else:
                            next_edge = next(iter(sorted(self.P_edges - visited_edges_keys)))
                            if self.recursive_ullman(next_edge, candidate_copy):
                                return True
                
                        # Remove mapping if this branch fails (backtrack)
                        del self.visited_edges[e]
                        del self.visited_nodes[v]

                if start:
                    self.failure_count += 1
                    if self.failure_count%100 == 0:
                        print(f"\rUllman failures: {self.failure_count}        ", end="")
        # No valid mapping found
        
        return False

    def edge_equals(self, e1, e2):
        """
        Check if two edges are equal.
        
        Args:
            e1: First edge (tuple of vertices)
            e2: Second edge (tuple of vertices)
            
        Returns:
            bool: True if edges are equal, False otherwise
        """
        return (e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])
    
