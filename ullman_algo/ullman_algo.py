import networkx as nx
import numpy as np

class UllmanAlgorithm:
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
        if len(P.nodes()) > len(G.nodes()):
            raise ValueError("P cannot be larger than G")
            
        # Store degree information for each vertex
        self.P_dictionary_by_vertex = dict(P.degree())
        self.G_dictionary_by_vertex = dict(G.degree())
        self.G_vertices = set(G.nodes())
        self.P_vertices = set(P.nodes())
        
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
        
        self.visited = {}

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
                # vertex p in P can be mapped to vertex g in G if the degree of p <= degree of g
                if p_degree <= g_degree:
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
            for g, g_degree in self.G_dictionaxry_by_vertex.items():
                # vertex p in P can be mapped to vertex g in G if the degree of p <= degree of g
                if p_degree == g_degree:
                    candidate_mappings[p_node_to_index.get(p)][g_node_to_index.get(g)] = True
        return candidate_mappings


    def get_unmapped_vertices(self):
        """
        Get vertices in G that were not mapped to any vertex in P.
        
        Returns:
            set: Vertices in G not included in the mapping
        """
        return set(self.G_vertices) - set(self.visited.values())
    
    def get_mapping(self):
        """
        Get the current mapping of vertices from P to G.
        
        Returns:
            dict: Mapping of vertices in P to vertices in G
        """
        return self.visited

    def ullman(self, exact_match):
        """
        Execute Ullman's algorithm to find if P is a subgraph of G.

        Parameters:
            exact_match (bool): If True, find an exact match (same degree), otherwise find a subgraph match
        
        Returns:
            bool: True if P is a subgraph of G, False otherwise
        """
        if len(self.adj_list_P) == 0:
            return True
        first_vertex = next(iter(self.adj_list_P.keys()))
        if exact_match:
            candidate_mapping_matrix = self.exact_candidate_mappings()
        else:
            candidate_mapping_matrix = self.candidate_mappings()
        return self.recursive_ullman(first_vertex, candidate_mapping_matrix, self.visited)

    def recursive_ullman(self, x, candidate_mapping_matrix, visited):
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
        x_idx = self.p_node_to_index[x]
        
        # Base case: if all nodes are visited
        if len(visited) == len(self.adj_list_P):
            return True
            
        visited_nodes = set(visited.keys())
        x_neighbors = set(self.adj_list_P[x])
        unvisited_neighbors = x_neighbors - visited_nodes
        
        # Get possible matches using matrix index
        possible_matches = np.where(candidate_mapping_matrix[x_idx])[0]

        # Try each possible match for vertex x
        for a_idx in possible_matches:
            a = self.g_index_to_node[a_idx]
            a_neighbors = set(self.adj_list_G[a])
            
            # Check if all unvisited neighbors of x can be matched to neighbors of a
            if all(any(candidate_mapping_matrix[self.p_node_to_index[unvisited_neighbor]][self.g_node_to_index[a_neighbor]] 
                        for a_neighbor in a_neighbors) 
                   for unvisited_neighbor in unvisited_neighbors):
                
                # Create a new constraint matrix
                candidate_copy = np.copy(candidate_mapping_matrix)
                candidate_copy[:, a_idx] = False  # Mark vertex a as used
                
                # Apply constraints: neighbors of x can only match to neighbors of a
                for unvisited_neighbor in unvisited_neighbors:
                    unvisited_idx = self.p_node_to_index[unvisited_neighbor]
                    for i in range(len(self.g_node_to_index)):
                        if self.g_index_to_node[i] not in a_neighbors:
                            candidate_copy[unvisited_idx][i] = False
                
                # Add current mapping
                visited[x] = a
                
                # Choose next vertex to match (prioritize neighbors)
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
                
                # Remove mapping if this branch fails (backtrack)
                del visited[x]
                
        return False

