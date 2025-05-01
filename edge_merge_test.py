import networkx as nx
from ullman_algo import UllmanAlgorithm
from collections import defaultdict

def edge_based_merge(G, P):
    """
    Merges two graphs based on edge-based subgraph isomorphism.

    This function attempts to merge two size-k graphs, `G` and `P`, by checking
    if they share exactly (k-1) edges. If they do, it merges them into a new
    graph containing exactly (k+1) edges. The isomorphism of (k-1)-edge subgraphs
    is checked using Ullman's algorithm to ensure a valid join.

    Args:
        G (nx.Graph): The first graph to be merged.
        P (nx.Graph): The second graph to be merged.

    Returns:
        list: A list containing the merged graph if valid, or an empty list if not.
    """
    print("/-------Entering edge_based_merge-------/")
    # Ensure the two graphs have the same edge count
    if G.number_of_edges() != P.number_of_edges():
        return []
    
    if G.number_of_edges() == 1 and P.number_of_edges() == 1: # K=1 Case
        return k1_join(G, P)

    merged_results = []

    # Loop through all possible k-1 subgraphs and check for isomorphism
    for u_p, v_p in P.edges():
        P_rem = nx.Graph(P)
        P_rem.remove_edge(u_p, v_p)

        # Avoid checking isomorphisms for 0-degree nodes
        for node in (u_p, v_p):
            if P_rem.degree(node) == 0:
                P_rem.remove_node(node)
    
        # Try removing this P-edge against each G-edge
        for u_g, v_g in G.edges():
            G_rem = nx.Graph(G)
            G_rem.remove_edge(u_g, v_g)

            for node in (u_p, v_p): 
                if G_rem.degree(node) == 0: # Avoid checking isomorphisms for 0-degree nodes
                    G_rem.remove_node(node)

            # Check if the remaining "root" size k-1 graph is a subgraph of G.
            # If it is, we can merge the two graphs.
            iso = UllmanAlgorithm(G_rem, P_rem)
            if not iso.ullman(False):
                continue
            mapping = iso.get_mapping()
            print(f"[DEBUG] Found join: P-edge=({u_p},{v_p}) - G-edge=({u_g},{v_g}); mapping={mapping}")


            # We want u_p to be the "anchor"  (degree > 1), 
            # and v_p to be the "leaf" (degree == 1)
            # This is for consistency, we need to know which node to refer to
            if G.degree(u_g) >= G.degree(v_g):
                g_leaf = v_g
            else:
                g_leaf = u_g

            if P.degree(u_p) >= P.degree(v_p):
                p_leaf = v_p
                p_anchor = u_p
            else:
                p_leaf = u_p
                p_anchor = v_p
            
            p_leaf_label = P.nodes[p_leaf].get('label')
            g_leaf_label = G.nodes[g_leaf].get('label')

            
            # /-----Candidate 1-----/
            # attach the P-leaf (node) along with its edge itself to G
            cand1 = nx.Graph(G)
            p_node = max(G.nodes()) + 1
            # hook it up to the mapped anchor
            cand1.add_node(p_node, label=p_leaf_label)
            cand1.add_edge(mapping[p_anchor], p_node)
            merged_results.append(cand1)

            # /-----Candidate 2-----/
            # add back the join edge between mapped nodes (only if labels match)
            # get the two labels
            print(f'G label {g_leaf_label}')
            print(f'P label {p_leaf_label}')
            print(f'P u P {u_p}')
            print(f'P v P {v_p}')
    
            if p_leaf_label == g_leaf_label:
                cand2 = nx.Graph(G)
                cand2.add_edge(mapping[p_anchor], p_leaf)
                merged_results.append(cand2)

    print(f"[DEBUG] edge_based_merge generated {len(merged_results)} candidates\n")
    return merged_results

def k1_join(G, P):
    """
    Given two single-edge graphs G and P, extend them by joining on their shared vertex label
    and returning the 2-edge path.

    ex. A-B + B-C -> A-B-C

    Returns:
        list: a list with exactly one merged graph (or [] if they don't share exactly one label).
    """

    merged_results = []
    labels_G = nx.get_node_attributes(G, 'label')
    labels_P = nx.get_node_attributes(P, 'label')

    shared_labels = set(labels_G.values()) & set(labels_P.values()) # Get intersection of node with the same labels
    if len(shared_labels) != 1:
        return merged_results
    label = shared_labels.pop()

    # Get the shared vertex
    g_join = next(n for n, l in labels_G.items() if l == label)
    p_join = next(n for n, l in labels_P.items() if l == label)

    # Get the neighbor of the shared vertex
    p_neighbor = next(n for n in P.neighbors(p_join))



    new_node = max(G.nodes()) + 1 # just pick an ID that doesn't cause collision with other IDs

    # Create k=1 candidate by adding P's node (shared label)'s neighbor with G's node (shared label)
    cand = nx.Graph()
    cand.add_nodes_from(G.nodes(data=True))
    cand.add_edges_from(G.edges())
    cand.add_node(new_node, label=labels_P[p_neighbor])
    cand.add_edge(g_join, new_node)
    merged_results.append(cand)
    return merged_results

    

def main():
    G = nx.Graph()
    G.add_node(0, label='A')
    G.add_node(1, label='A')
    G.add_node(2, label='B')
    G.add_node(3, label='C')
    G.add_edges_from([(0, 1), (1, 2), (0,2), (1,3)])

    # Graph P: 2–5–6
    P = nx.Graph()
    P.add_node(0, label='A')
    P.add_node(1, label='A')
    P.add_node(2, label='B')
    P.add_node(3, label='C')
    P.add_edges_from([(0, 1), (1, 2), (0,2), (2,3)])

    print("\n--- Input Graphs ---")
    print("G nodes:", list(G.nodes(data=True)))
    print("G edges:", list(G.edges()))
    print("P nodes:", list(P.nodes(data=True)))
    print("P edges:", list(P.edges()))

    # Run your merge
    merged = edge_based_merge(G, P)

    print(f"\n--- {len(merged)} Merged Candidate(s) ---")
    for i, M in enumerate(merged):
        print(f"Candidate {i}:")
        print("  nodes:", list(M.nodes(data=True)))
        print("  edges:", list(M.edges()))

if __name__ == "__main__":
    main()


