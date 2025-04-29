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

    merged_results = []

    # Only consider the P-edge that does not already exist in G
    for u_p, v_p in P.edges():
        if G.has_edge(u_p, v_p):
            continue

        # e want u_p to be the anchor (degree > 1 in P), 
        # and v_p to be the leaf (degree == 1 in P)
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
            if not iso.ullman(exact_match=True):
                continue

            mapping = iso.get_mapping()
            print(f"[DEBUG] Found join: P-edge=({u_p},{v_p}) - G-edge=({u_g},{v_g}); mapping={mapping}")

            # Candidate 1: add back the join edge between mapped nodes
            mu, mv = mapping[u_p], mapping[v_p]
            cand1 = nx.Graph(G)
            if not cand1.has_edge(mu, mv):
                cand1.add_edge(mu, mv)
                merged_results.append(cand1)
                print(f"[DEBUG] Cand1 edges: {sorted(cand1.edges())}")

            # Candidate 2: attach a fresh node at the mapped u_p
            new_node = max(G.nodes()) + 1
            cand2 = nx.Graph()
            cand2.add_nodes_from(G.nodes(data=True))
            cand2.add_edges_from(G.edges(data=True))
            label_v = P.nodes[v_p].get('label')
            cand2.add_node(new_node, label=label_v)
            cand2.add_edge(mapping[u_p], new_node)
            merged_results.append(cand2)
            print(f"[DEBUG] Cand2 edges: {sorted(cand2.edges())}")

            # We only want the two candidates for the one differing edge
            return merged_results

    print(f"[DEBUG] edge_based_merge generated {len(merged_results)} candidates\n")
    return merged_results



def main():
    # Define three nodes with labels X, Y, Z
    labels = {1: 'X', 2: 'Y', 3: 'Z'}

    # Graph 1: x–y–z path
    G1 = nx.Graph()
    G1.add_nodes_from(labels.keys())
    nx.set_node_attributes(G1, labels, 'label')
    G1.add_edges_from([(1, 2), (2, 3)])

    # Graph 2: y–z–x path
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
    
     # Define four nodes: A1, A2, B, C as integers 1,2,3,4
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

if __name__ == "__main__":
    main()


