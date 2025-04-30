import networkx as nx
from ullman_algo import UllmanAlgorithm
import math

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
    print("\n[DEBUG] Entering edge_based_merge")
    # Ensure the two graphs have the same edge count
    if G.number_of_edges() != P.number_of_edges():
        print(f"[DEBUG] Edge count mismatch: G={G.number_of_edges()}, P={P.number_of_edges()}")
        return []
    
    if G.number_of_edges() == 1 and P.number_of_edges() == 1:
        return k1_join(G, P)

    merged_results = []

    # Only consider the P-edge that does not already exist in G
    for u_p, v_p in P.edges():
        if G.has_edge(u_p, v_p):
            continue

        # We want u_p to be the anchor (degree > 1), 
        # and v_p to be the leaf (degree == 1)
        # This is for consistency, because later when we define a fresh node
        # we need to know which node (label) to refer
        if P.degree(u_p) < P.degree(v_p):
            u_p, v_p = v_p, u_p

        # Try removing this P-edge against each G-edge
        for u_g, v_g in G.edges():
            P_rem = nx.Graph(P)
            P_rem.remove_edge(u_p, v_p)
            G_rem = nx.Graph(G)
            G_rem.remove_edge(u_g, v_g)

            # Exact-match isomorphism on the k-1 graphs
            iso = UllmanAlgorithm(G_rem, P_rem)
            if not iso.ullman(exact_match=False):
                continue

            mapping = iso.get_mapping()
            print(f"[DEBUG] Found join: P-edge=({u_p},{v_p}) - G-edge=({u_g},{v_g}); mapping={mapping}")

            # Candidate 1: add back the join edge between mapped nodes (only if labels match)
           
            if G.degree(u_g) >= G.degree(v_g):
                g_leaf = v_g
            else:
                g_leaf = u_g

            # get the two labels
            p_leaf_label = P.nodes[v_p].get('label')
            g_leaf_label = G.nodes[g_leaf].get('label')

    
            if p_leaf_label == g_leaf_label:
                mu, mv = mapping[u_p], mapping[v_p]
                cand1 = nx.Graph(G)
                if not cand1.has_edge(mu, mv):
                    cand1.add_edge(mu, mv)
                    merged_results.append(cand1)
                    print(f"[DEBUG] Cand1 edges: {sorted(cand1.edges())}")

            # Candidate 2: attach the P-leaf (node) along with its edge itself to G
            leaf = v_p
            leaf_label = P.nodes[leaf]['label']

            cand2 = nx.Graph(G)
            # only add the leaf node if it's not already in G
            if leaf not in cand2.nodes():
                cand2.add_node(leaf, label=leaf_label)

            # hook it up to the mapped anchor
            cand2.add_edge(mapping[u_p], leaf)
            merged_results.append(cand2)
            print(f"[DEBUG] Cand2 edges: {sorted(cand2.edges())}")
            print(f"[DEBUG] Cand2 edges: {sorted(cand2.edges())}")

            # We only want the two candidates for the one differing edge

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
    # Define three nodes with labels X, Y, Z
    labels = {1: 'X', 2: 'Y', 3: 'Z'}

    # Graph 1: x-y-z path
    G1 = nx.Graph()
    G1.add_nodes_from(labels.keys())
    nx.set_node_attributes(G1, labels, 'label')
    G1.add_edges_from([(1, 2), (2, 3)])

    # Graph 2: y-z-x path
    G2 = nx.Graph()
    G2.add_nodes_from(labels.keys())
    nx.set_node_attributes(G2, labels, 'label')
    G2.add_edges_from([(2, 3), (3, 1)])

    print("G1 edges:", sorted(G1.edges()), "labels:", labels)
    print("G2 edges:", sorted(G2.edges()), "labels:", labels)

    # Perform edge-based merge
    merged = edge_based_merge(G1, G2)
    print("\nMerged candidates:")
    if not merged:
        print("  No candidates generated.")
    for i, m in enumerate(merged, 1):
        nodes = [(n, m.nodes[n]['label']) for n in sorted(m.nodes())]
        print(f"  Candidate {i}:")
        print("    Nodes:", nodes)
        print("    Edges:", sorted(m.edges()))
    

    labels = {1: 'A', 2: 'A', 3: 'B', 4: 'C'}

    # Graph P1: edges (A1-A2), (A1-B), (A2-B), (A2-C)
    P1 = nx.Graph()
    P1.add_nodes_from(labels.keys())
    nx.set_node_attributes(P1, labels, 'label')
    P1.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4)])

    # Graph P2: edges (A1-A2), (A1-B), (A2-B), (B-C)
    P2 = nx.Graph()
    P2.add_nodes_from(labels.keys())
    nx.set_node_attributes(P2, labels, 'label')
    P2.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])

    print("\n")
    print("Second Test")
    print("\n")
    print("P1 edges:", sorted(P1.edges()), "with labels:", [(n, P1.nodes[n]['label']) for n in P1.nodes()])
    print("P2 edges:", sorted(P2.edges()), "with labels:", [(n, P2.nodes[n]['label']) for n in P2.nodes()])

    # Perform edge-based merge
    merged = edge_based_merge(P1, P2)
    print("\nMerged candidates:")
    if not merged:
        print("  No candidates generated.")
    for i, m in enumerate(merged, 1):
        nodes = [(n, m.nodes[n]['label']) for n in sorted(m.nodes())]
        print(f"  Candidate {i}:")
        print("    Nodes:", nodes)
        print("    Edges:", sorted(m.edges()))
  # ─────── Single-edge merge: A-B with B-C ───────
    # Use disjoint node sets so mapping is unambiguous
    labels_ab = {1: 'A', 2: 'B'}
    G_ab = nx.Graph()
    G_ab.add_nodes_from(labels_ab)
    nx.set_node_attributes(G_ab, labels_ab, 'label')
    G_ab.add_edge(1, 2)

    labels_bc = {3: 'B', 4: 'C'}
    G_bc = nx.Graph()
    G_bc.add_nodes_from(labels_bc)
    nx.set_node_attributes(G_bc, labels_bc, 'label')
    G_bc.add_edge(3, 4)

    print("\nSingle-edge merge test (A-B with B-C)")
    print("G_ab edges:", sorted(G_ab.edges()), "labels:", [(n, G_ab.nodes[n]['label']) for n in G_ab])
    print("G_bc edges:", sorted(G_bc.edges()), "labels:", [(n, G_bc.nodes[n]['label']) for n in G_bc])

    merged_single = edge_based_merge(G_ab, G_bc)
    print("\nMerged candidates:")
    if not merged_single:
        print("  No candidates generated.")
    else:
        for i, m in enumerate(merged_single, 1):
            nodes = [(n, m.nodes[n]['label']) for n in sorted(m.nodes())]
            print(f"  Candidate {i}:")
            print("    Nodes:", nodes)
            print("    Edges:", sorted(m.edges()))

if __name__ == "__main__":
    main()


