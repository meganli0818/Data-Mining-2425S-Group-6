import networkx as nx
import matplotlib.pyplot as plt
import time
from ullman_algo.ullman_algo_edge import UllmanAlgorithm

def create_labeled_graph(edges, labels=None):
    """Create a graph with labeled nodes."""
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    if labels:
        nx.set_node_attributes(G, labels, 'label')
    else:
        # Default labels (all nodes get the same label)
        default_labels = {node: 'default' for node in G.nodes()}
        nx.set_node_attributes(G, default_labels, 'label')
    
    return G

def visualize_graph(G, title="Graph"):
    """Visualize a graph with node labels."""
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold')
    
    # Add edge labels
    edge_labels = {(u, v): f"{u}-{v}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Add node labels from label attribute
    node_labels = nx.get_node_attributes(G, 'label')
    label_pos = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels={n: f"({node_labels[n]})" for n in G.nodes()})
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

def test_basic_subgraph_isomorphism():
    """Test basic subgraph isomorphism with simple graphs."""
    print("\n=== Basic Subgraph Isomorphism Test (Expect Match Found) ===")
    
    # Create a larger graph G
    G_edges = [(1, 2), (2, 3), (3, 4), (4, 1), (2, 4), (3, 5)]
    G_labels = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'A'}
    G = create_labeled_graph(G_edges, G_labels)
    
    # Create a pattern graph P (a triangle)
    P_edges = [(1, 2), (2, 3), (3, 1)]
    P_labels = {1: 'A', 2: 'B', 3: 'C'}
    P = create_labeled_graph(P_edges, P_labels)
    
    visualize_graph(G, "Larger Graph G")
    visualize_graph(P, "Pattern Graph P")
    
    # Test Ullman algorithm
    start_time = time.time()
    ullman = UllmanAlgorithm(G, P)
    result = ullman.ullman(exact_match=False)
    end_time = time.time()
    
    print(f"Isomorphism found: {result}")
    if result:
        print("Mapping:", ullman.visited_nodes)
        print("Edge mapping:", ullman.visited_edges)
    print(f"Time taken: {end_time - start_time:.6f} seconds")

def test_exact_match():
    """Test exact graph isomorphism (same structure and degrees)."""
    print("\n=== Exact Match Test (Expect Match Found) ===")
    
    # Create two isomorphic graphs with different node IDs
    G_edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    G_labels = {1: 'A', 2: 'B', 3: 'A', 4: 'B'}
    G = create_labeled_graph(G_edges, G_labels)
    
    P_edges = [(10, 20), (20, 30), (30, 40), (40, 10)]
    P_labels = {10: 'A', 20: 'B', 30: 'A', 40: 'B'}
    P = create_labeled_graph(P_edges, P_labels)
    
    visualize_graph(G, "Graph G")
    visualize_graph(P, "Graph P (Same Structure)")
    
    # Test Ullman algorithm with exact matching
    start_time = time.time()
    ullman = UllmanAlgorithm(G, P)
    result = ullman.ullman(exact_match=True)
    end_time = time.time()
    
    print(f"Exact isomorphism found: {result}")
    if result:
        print("Mapping:", ullman.visited_nodes)
        print("Edge mapping:", ullman.visited_edges)
    print(f"Time taken: {end_time - start_time:.6f} seconds")

def test_no_isomorphism():
    """Test when no isomorphism exists."""
    print("\n=== No Isomorphism Test (Expect No Match Found) ===")
    
    # Create a larger graph G
    G_edges = [(1, 2), (2, 3), (3, 1)]
    G_labels = {1: 'A', 2: 'B', 3: 'C'}
    G = create_labeled_graph(G_edges, G_labels)
    
    # Create a pattern graph P with a different structure
    P_edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    P_labels = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    P = create_labeled_graph(P_edges, P_labels)
    
    visualize_graph(G, "Graph G")
    visualize_graph(P, "Pattern Graph P (No Match)")
    
    # Test Ullman algorithm
    start_time = time.time()
    ullman = UllmanAlgorithm(G, P)
    result = ullman.ullman(exact_match=False)
    end_time = time.time()
    
    print(f"Isomorphism found: {result}")
    print(f"Time taken: {end_time - start_time:.6f} seconds")

def test_label_constraints():
    """Test isomorphism with label constraints."""
    print("\n=== Label Constraints Test (Expect No Match Found) ===")
    
    # Create a larger graph G
    G_edges = [(1, 2), (2, 3), (3, 4), (4, 1), (2, 4), (3, 5)]
    G_labels = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
    G = create_labeled_graph(G_edges, G_labels)
    
    # Create a pattern graph P
    P_edges = [(1, 2), (2, 3), (3, 1)]
    P_labels = {1: 'A', 2: 'B', 3: 'C'}
    P = create_labeled_graph(P_edges, P_labels)
    
    visualize_graph(G, "Graph G with Labels")
    visualize_graph(P, "Pattern Graph P with Labels")
    
    # Test Ullman algorithm
    start_time = time.time()
    ullman = UllmanAlgorithm(G, P)
    result = ullman.ullman(exact_match=False)
    end_time = time.time()
    
    print(f"Isomorphism found: {result}")
    if result:
        print("Mapping:", ullman.visited_nodes)
        print("Edge mapping:", ullman.visited_edges)
    print(f"Time taken: {end_time - start_time:.6f} seconds")

def test_larger_graph():
    """Test with a larger graph to evaluate performance."""
    print("\n=== Larger Graph Test (Expect Match Found)===")
    
    # Create a larger grid graph
    G = nx.grid_2d_graph(4, 4)
    # Convert to integer node labels
    G = nx.convert_node_labels_to_integers(G)
    # Add labels
    labels = {node: 'X' if node % 2 == 0 else 'Y' for node in G.nodes()}
    nx.set_node_attributes(G, labels, 'label')
    
    # Create a small pattern (a path)
    P = nx.path_graph(3)
    # Add labels
    p_labels = {0: 'X', 1: 'Y', 2: 'X'}
    nx.set_node_attributes(P, p_labels, 'label')
    
    visualize_graph(G, "Larger Grid Graph G")
    visualize_graph(P, "Pattern Path Graph P")
    
    # Test Ullman algorithm
    start_time = time.time()
    ullman = UllmanAlgorithm(G, P)
    result = ullman.ullman(exact_match=False)
    end_time = time.time()
    
    print(f"Isomorphism found: {result}")
    if result:
        print("Mapping:", ullman.visited_nodes)
        print("First few edge mappings:", {k: v for i, (k, v) in enumerate(ullman.visited_edges.items()) if i < 5})
    print(f"Time taken: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    print("Testing Edge-Based Ullman's Algorithm\n")
    
    #Run tests
    test_basic_subgraph_isomorphism()
    test_exact_match()
    test_no_isomorphism()
    test_label_constraints()
    test_larger_graph()


    
    print("\nAll tests completed!")

    # plt.show()

