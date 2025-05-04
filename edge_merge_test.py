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
        return k1_join(G, P)

    merged_results = []

    # Loop through all edges in P, removing one at a time.
    for u_p, v_p in P.edges():
        P_rem = nx.Graph(P)
        P_rem.remove_edge(u_p,v_p)
        iso = UllmanAlgorithmEdge(G, P_rem)

        # Check if the remaining "root" size k-1 P-graph is a subgraph of G.
        # If it is, we can merge the two graphs.
        print(iso.ullman(False))
        if iso.ullman(False):
            unmapped_edges_g = iso.get_unmapped_edges_in_G()
            G_rem = nx.Graph(G)
            for unmapped_edge_g in unmapped_edges_g:
                G_rem.remove_edge(*unmapped_edge_g)
            print(f"UNMAPPED EGES G:  {(unmapped_edges_g)}")
            exact_match = UllmanAlgorithmEdge(G_rem, P_rem)
            print("G rem nodes:", list(G_rem.nodes(data=True)))
            print("G rem edges:", list(G_rem.edges(data=True)))
            print("P rem nodes:", list(P_rem.nodes(data=True)))
            print("P rem edges:", list(P_rem.edges(data=True)))
            print("\n")
            
            print(f"EXACT MATCH {exact_match.ullman(True)}")
            print("\n")
            if exact_match.ullman(True):

                unmapped_p_nodes = iso.get_unmapped_vertices_in_P()
                unmapped_g_nodes = iso.get_unmapped_vertices_in_G()
                print(list(unmapped_p_nodes))
                for unmapped_node in unmapped_p_nodes:
                    print(f" P NODESSS: {P.nodes[unmapped_node]['label']}")
                print(list(unmapped_g_nodes))
                for unmapped_node in unmapped_p_nodes:
                    print(F"G NODESSS : {G.nodes[unmapped_node]['label']}")
                
                unmapped_edge_p = (u_p, v_p)

                # /----- Create new merged graph using G as base -----/
                merged_graph = nx.Graph(G)

                # The case where P (to be added to G) does not have an unmapped node
                # // Weird nx behavior about implicitly adding any nodes that don’t already exist when defining edges
                if(len(unmapped_p_nodes) == 0):
                    merged_graph.add_edge(*unmapped_edge_p)
                    merged_results.append(merged_graph)
                    print("RETURNED 1")

                # Using vertex mapping dictionary:
                # 1. Find mapping of u, v to G
                # 2. If these do not map to anything in G, we create new nodes.
                # 3. Then, we add define the edge between new node and merged graph.
                

                mapping = iso.get_mapping()

                if any(node not in mapping for node in (u_p, v_p)): 
                    print("CANDIDATE 1 TIME")
                    print(f"Up {u_p}")
                    print(f"Vp {v_p}")
                    # /--- Candidate 1 ---/
                    # If P_node is missing in G, create node and add to merged graph
                    
                    for unmapped_node_p in unmapped_p_nodes:
                        
                        print("M nodes:", list(merged_graph.nodes(data=True)))
                        print("M edges:", list(merged_graph.edges(data=True)))
                    
                        new_node = max(G.nodes()) + 1 if G.nodes() else 1
                        merged_graph.add_node(new_node, label=P.nodes[unmapped_node_p]['label'])
                        print("M nodes after add :", list(merged_graph.nodes(data=True)))
                        print("M edges:", list(merged_graph.edges(data=True)))
                        print(f"LABEL OF NEW NODE {merged_graph.nodes[new_node]['label']}")
                        existing_node = u_p if u_p in mapping else v_p
                        print(merged_graph.nodes[existing_node]['label'])
                        print(merged_graph.edges)
                        merged_graph.add_edge(new_node, mapping[existing_node])
                        print(merged_graph.edges)
                        print(f"MAPPING {merged_graph.nodes[mapping[existing_node]]['label']}")
                        merged_results.append(merged_graph)
                    
                    # /--- Candidate 2 ---/
                    # Check if unmapped node p and unmapped node g are equal 

                        for unmapped_node_g in unmapped_g_nodes:

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
    G.add_node(0, label='2')
    G.add_node(1, label='5')
    G.add_node(2, label='2')
    G.add_node(3, label='6')
    G.add_edges_from([(0, 1), (0,2), (1,3)])

    P = nx.Graph()
    P.add_node(0, label='2')
    P.add_node(1, label='5')
    P.add_node(2, label='2')
    P.add_node(3, label='6')
    P.add_edges_from([(0, 1), (0,2), (1,3)])

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
