import networkx as nx
from ullman_algo import UllmanAlgorithmEdge
from collections import defaultdict
from edge_apriori import edge_based_merge



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

    shared_labels = sorted(set((labels_G.values())) & set((labels_P.values()))) # Get intersection of node with the same labels
    print(list(shared_labels))
    if len(shared_labels) < 1:
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
    G.add_node(0, label='6')
    G.add_node(1, label='5')
    G.add_edges_from([(0, 1)])

    P = nx.Graph()
    P.add_node(0, label='6')
    P.add_node(1, label='5')
    P.add_edges_from([(0, 1)])

    print("\n--- Input Graphs ---")
    print("G nodes:", list(G.nodes(data=True)))
    print("G edges:", list(G.edges()))
    print("P nodes:", list(P.nodes(data=True)))
    print("P edges:", list(P.edges()))
    print("END")
    print("\n")

    # Run merge
    merged = edge_based_merge(G, P)

    print(f"\n--- {len(merged)} Merged Candidate(s) ---")
    for i, M in enumerate(merged):
        print(f"Candidate {i}:")
        print("  nodes:", list(M.nodes(data=True)))
        print("  edges:", list(M.edges()))

if __name__ == "__main__":
    main()
