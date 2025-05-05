import networkx as nx
from ullman_algo_edge import UllmanAlgorithmEdge
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
    # Ensure the two graphs have the same edge count
    if G.number_of_edges() != P.number_of_edges():
        return []
    
    if G.number_of_edges() == 1 and P.number_of_edges() == 1: # K=1 Case
        print("[DEBUG] K==1 CASE ENTERED")
        return k1_join(G, P)

    merged_results = []

    # Loop through all edges in P, removing one at a time.
    for u_p, v_p in sorted(P.edges()):
        P_rem = nx.Graph(P)
        P_rem.remove_edge(u_p,v_p)
        iso = UllmanAlgorithmEdge(G, P_rem)

        # Check if the remaining "root" size k-1 P-graph is a subgraph of G.
        # If it is, we can merge the two graphs.
        print(iso.ullman(False))
        if iso.ullman(False):
            unmapped_edges_g = iso.get_unmapped_edges_in_G()
            G_rem = nx.Graph(G)
            for unmapped_edge_g in sorted(unmapped_edges_g):
                G_rem.remove_edge(*unmapped_edge_g)

            exact_match = UllmanAlgorithmEdge(G_rem, P_rem)
          
            if exact_match.ullman(True):

                unmapped_p_nodes = iso.get_unmapped_vertices_in_P()
                unmapped_g_nodes = iso.get_unmapped_vertices_in_G()

            
                mapping = iso.get_mapping()
                
                unmapped_edge_p = (u_p, v_p)
                merged_graph = nx.Graph(G)
            
                if (node not in mapping for node in (u_p, v_p)):

                    if (u_p in mapping):
                        existing_node_p = u_p
                        unmapped_node_p = v_p
                    else:
                        existing_node_p = v_p
                        unmapped_node_p = u_p
                    new_node = max(G.nodes()) + 1 if G.nodes() else 1
                    merged_graph.add_node(new_node, label=P.nodes[unmapped_node_p]['label'])
                    existing_node = u_p if u_p in mapping else v_p
                    merged_graph.add_edge(new_node, mapping[existing_node_p])
                    merged_results.append(merged_graph)
                else:
      
                    #p doesnt havae unmapped node
                    if not merged_graph.has_edge(mapping[u_p], mapping[v_p]):
                        merged_graph.add_edge(mapping[u_p], mapping[v_p])
                        merged_results.append(merged_graph)

   
                    unmapped_node_g = next(iter(sorted(unmapped_g_nodes)))
                    if merged_graph.nodes[new_node]['label'] == G.nodes[unmapped_node_g]['label'] and not G.has_edge(unmapped_node_g, mapping[existing_node]): 
                        merged_graph2 = nx.Graph(G)
                        merged_graph2.add_edge(unmapped_node_g, mapping[existing_node])
                        merged_results.append(merged_graph2)
                        return merged_results
                return merged_results
            

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
    G.add_node(2, label='2')
    G.add_edges_from([(0, 1), (1,2)])

    P = nx.Graph()
    P.add_node(0, label='6')
    P.add_node(1, label='5')
    P.add_node(2, label='2')
    P.add_edges_from([(0, 1), (1,2)])

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
