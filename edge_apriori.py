import networkx as nx
from ullman_algo_edge import UllmanAlgorithmEdge
import math
import sys

# Debug flag to control output verbosity
DEBUG = True  # Set to False for production mode


def debug_print(*args, **kwargs):
    """Print only if DEBUG is True."""
    if DEBUG:
        print(*args, **kwargs)


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

            exact_match = UllmanAlgorithmEdge(G_rem, P_rem)
            
           
            if exact_match.ullman(True):

                unmapped_p_nodes = iso.get_unmapped_vertices_in_P()
                unmapped_g_nodes = iso.get_unmapped_vertices_in_G()
                print(list(unmapped_p_nodes))
                for unmapped_node in unmapped_p_nodes:
                    print(P.nodes[unmapped_node]['label'])
                print(list(unmapped_g_nodes))
                for unmapped_node in unmapped_p_nodes:
                    print(G.nodes[unmapped_node]['label'])
                
                unmapped_edge_p = (u_p, v_p)

                # /----- Create new merged graph using G as base -----/
                merged_graph = nx.Graph(G)

                # The case where P (to be added to G) does not have an unmapped node
                # // Weird nx behavior about implicitly adding any nodes that don’t already exist when defining edges
                if(len(unmapped_p_nodes) == 0):
                    merged_graph.add_edge(*unmapped_edge_p)
                    merged_results.append(merged_graph)
                    return merged_results

                # Using vertex mapping dictionary:
                # 1. Find mapping of u, v to G
                # 2. If these do not map to anything in G, we create new nodes.
                # 3. Then, we add define the edge between new node and merged graph.
                

                mapping = iso.get_mapping()

                if any(node not in mapping for node in (u_p, v_p)): 
                    print(f"Up {u_p}")
                    print(f"Vp {v_p}")
                    # /--- Candidate 1 ---/
                    # If P_node is missing in G, create node and add to merged graph
                    for unmapped_node_p in unmapped_p_nodes:
                       
                    
                        new_node = max(G.nodes()) + 1 if G.nodes() else 1
                        merged_graph.add_node(new_node, label=P.nodes[unmapped_node_p]['label'])
                      
                        existing_node = u_p if u_p in mapping else v_p
                        
                        merged_graph.add_edge(new_node, mapping[existing_node])
                        
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



def generate_candidates(freq_subgraphs):
    """
    Generates candidate subgraphs of size k+1 from a set of frequent subgraphs of size k.

    This function takes a list of frequent subgraphs of size k and generates 
    candidate subgraphs of size k+1 by edge-based merging every pair of frequent subgraphs. 
    It ensures that duplicate candidates are not added by checking for 
    isomorphism using the Ullman algorithm.

    Args:
        freq_subgraphs (list): A list or set of frequent subgraphs of size k.

    Returns:
        set: A set of candidate subgraphs of size k+1. If no candidates can 
             be generated or the input is empty, returns `None`.
    """
    if freq_subgraphs is None or len(freq_subgraphs) == 0:
        return None
    freq_subgraphs_list = list(freq_subgraphs)
    candidates = set()
    
    # Loop through all pairs of frequent subgraphs, merging each pair to create new candidates.
    for i in range(len(freq_subgraphs_list)):
        for j in range(i, len(freq_subgraphs_list)):
            new_candidates = edge_based_merge(freq_subgraphs_list[i], freq_subgraphs_list[j])
            if new_candidates is not None:
        
                # Check if each candidate is already generated.
                for new_candidate in new_candidates:
                    if new_candidate.number_of_edges() <= freq_subgraphs_list[i].number_of_edges() or new_candidate.number_of_edges() <= freq_subgraphs_list[j].number_of_edges():
                        print_graph_nodes_simple([new_candidate, freq_subgraphs_list[i], freq_subgraphs_list[j]])
                        print("ERROR")
                        sys.exit(1)
                    candidate_already_generated = False
                    for existing_candidate in candidates:
                        ullman_exact = UllmanAlgorithmEdge(existing_candidate, new_candidate)
                        
                        # No need to add candidate if it is already generated.
                        if ullman_exact.ullman(True):
                            candidate_already_generated = True
                            break

                    #  Add candidate only if it is not already generated
                    if not candidate_already_generated and nx.is_connected(new_candidate):    
                        candidates.add(new_candidate)
                        #debug_print("candidate found")
        print(f"\rGenerated with graph {i}/{len(freq_subgraphs_list)}...", end="")

    print()
    return candidates


def all_subgraphs_frequent(candidate, freq_subgraphs):
    """
    Checks if all (k-1)-size subgraphs of a k-size candidate graph are frequent.

    This function iteratively removes each edge from the candidate graph to 
    generate all possible (k-1)-size subgraphs. It then checks if each of these 
    subgraphs is present in the list of frequent subgraphs using the Ullman 
    algorithm for isomorphism.

    Args:
        candidate (nx.Graph): The candidate graph of size k.
        freq_subgraphs (list): A list of frequent subgraphs of size k-1.

    Returns:
        bool: True if all (k-1)-size subgraphs of the candidate are frequent, 
              False otherwise.
    """
    for u, v in candidate.edges():
        sub = nx.Graph(candidate)
        sub.remove_edge(u, v)
        if not nx.is_connected(sub):
            continue
        freq = False
        for subgraph in freq_subgraphs:
            ullman = UllmanAlgorithmEdge(subgraph, sub)
            if ullman.ullman(True):
                freq = True
                break
        if not freq:
            return False
    return True


def prune(candidates, freq_subgraphs):
    """
    Prunes candidate subgraphs based on whether all their (k-1)-size subgraphs 
    are frequent.

    This function iterates through a set of candidate subgraphs and retains 
    only those for which all (k-1)-size subgraphs are present in the list of 
    frequent subgraphs. The check is performed using the `all_subgraphs_frequent` 
    function.

    Args:
        candidates (set): A set of candidate subgraphs of size k.
        freq_subgraphs (list): A list of frequent subgraphs of size k-1.

    Returns:
        set: A set of pruned candidate subgraphs that satisfy the condition 
             that all their (k-1)-size subgraphs are frequent.
    """
    pruned_candidates = set()
    for candidate in candidates:
        if all_subgraphs_frequent(candidate, freq_subgraphs):
            pruned_candidates.add(candidate)
    return pruned_candidates


def all_single_edge_graphs(graph_dataset):
    """
    Generates all single-edge graphs from a dataset of graphs.

    This function extracts edges from the input graph dataset 
    and creates single edge graphs for each unique edge.

    Args:
        graph_dataset (list): A list of NetworkX graph objects representing the dataset.

    Returns:
        list: A list of single edge graphs, each containing exactly one edge connecting two labeled nodes.
    """
    unique_edge_labels = set()
    single_edge_graphs = []

    for graph in graph_dataset:
        for u, v in graph.edges():
            # sort labels to avoid duplicates (A,B) vs (B,A)
            label_u = graph.nodes[u].get('label')
            label_v = graph.nodes[v].get('label')
            if label_u is None or label_v is None:
                continue
            pair = tuple(sorted((label_u, label_v)))
            unique_edge_labels.add(pair)

    debug_print("unique edge label pairs:", unique_edge_labels)

    for label_u, label_v in unique_edge_labels:
        G = nx.Graph()
        G.add_node(0, label=label_u)
        G.add_node(1, label=label_v)
        G.add_edge(0, 1)
        single_edge_graphs.append(G)

    return single_edge_graphs


def apriori(graph_dataset, min_freq, verbose=None):
    """
    Apriori algorithm to find frequent subgraphs in a dataset of graphs.

    This function implements the Apriori algorithm to mine frequent subgraphs 
    from a dataset of graphs. It starts by identifying single_edge graphs (graphs 
    with a single edge) and iteratively generates larger candidate 
    subgraphs to check.

    Args:
        graph_dataset (list): A list of NetworkX graph objects representing the dataset.
        min_freq (float): Minimum frequency threshold (between 0.0 and 1.0). 
                          A subgraph is considered frequent if it appears in 
                          a fraction of at least min_freq graphs.
        verbose (bool, optional): Overrides the global `DEBUG` setting. If `True`, 
                                  enables debug output. If `None`, uses the global 
                                  `DEBUG` value.

    Returns:
        list: A list of frequent subgraphs. Each subgraph is a NetworkX graph object.
    """
    # Use provided verbosity or fall back to global setting
    local_debug = DEBUG if verbose is None else verbose
    
    # Save original DEBUG value
    original_debug = globals()['DEBUG']
    globals()['DEBUG'] = local_debug
    
    min_support = math.ceil(min_freq * len(graph_dataset))
    freq_subgraphs = []

    # Generate all singletons
    single_edge_graphs = all_single_edge_graphs(graph_dataset)
    curr_freq_subgraphs = []
    for single_edge_graph in single_edge_graphs:
        # Count support for each singleton
        candidate_supp = {}
        for graph in graph_dataset:
            if single_edge_graph.number_of_edges() <= graph.number_of_edges():
                ullman = UllmanAlgorithmEdge(graph, single_edge_graph)
                if ullman.ullman(False):
                    if single_edge_graph not in candidate_supp:
                        candidate_supp[single_edge_graph] = 1
                    else:
                        candidate_supp[single_edge_graph] += 1
        # Save singletons based on minimum support
        for candidate, supp in candidate_supp.items():
            if supp >= min_support:
                curr_freq_subgraphs.append(candidate)
    
    debug_print("number of frequent single-edge graphs: ", len(curr_freq_subgraphs))
    debug_print("frequent single edge graphs ")
    print_graph_nodes_simple(curr_freq_subgraphs)

    # Apriori algorithm
    while curr_freq_subgraphs and len(curr_freq_subgraphs) > 0:
        
        # Generate candidates of size k+1 from current frequent subgraphs of size k
        freq_subgraphs.extend(curr_freq_subgraphs)
        unpruned_candidates = generate_candidates(curr_freq_subgraphs)
        print("generated candidates of size:", curr_freq_subgraphs[0].number_of_edges() + 1)
        debug_print("generated candidates: ")
        print_graph_nodes_simple(unpruned_candidates)

        # Prune candidates
        candidates = prune(unpruned_candidates, curr_freq_subgraphs)
        print("pruned candidates of size:", curr_freq_subgraphs[0].number_of_edges() + 1)
        print("number of candidates: ", len(candidates))

        
        print(f"size of K : {curr_freq_subgraphs[0].number_of_edges() + 1}")
        print(f"size of curr_freq_subgraph : {len(curr_freq_subgraphs)}")


        # Count support for each candidate
        candidate_supp = {}
        counter = 1
        for graph in graph_dataset:
            inner_counter = 1
            for candidate in candidates:
                if(curr_freq_subgraphs[0].number_of_edges() >= candidate.number_of_edges()):
                    print("ERROR! candidate has less edges than before")
                    print("candidate: ")
                    print_graph_nodes_simple([candidate])
                    print("old graph")
                    print_graph_nodes_simple([curr_freq_subgraphs[0]])
                    sys.exit(1)
                if candidate.number_of_edges() <= graph.number_of_edges():
                    ullman = UllmanAlgorithmEdge(graph, candidate)
                    print(f"\rChecked candidate {inner_counter}/{len(candidates)} with graph {counter}/{len(graph_dataset)}    ", end="")
                    if ullman.ullman(False):
                        if candidate not in candidate_supp:
                            candidate_supp[candidate] = 1
                        else:
                            candidate_supp[candidate] += 1
                inner_counter += 1
            counter += 1
        
        print("\nCalculated support of size:", curr_freq_subgraphs[0].number_of_nodes() + 1)
        debug_print("number of potential candidates: ", len(candidate_supp))
        
        # Save candidates based on minimum support for the next round
        curr_freq_subgraphs = []
        for candidate, supp in candidate_supp.items():
            if supp >= min_support:
                curr_freq_subgraphs.append(candidate)
        
        print("number of candidates: ", len(curr_freq_subgraphs))
        print()
    
    
    # Restore original DEBUG value
    globals()['DEBUG'] = original_debug
    
    return freq_subgraphs



def print_graph_nodes_simple(graph_list, debug_only=True):
    """
    Print the nodes of each graph along with their labels.
    
    Args:
        graph_list: List of NetworkX graph objects
        debug_only: If True, only print when DEBUG is True
    """
    if debug_only and not DEBUG:
        return
        
    all_nodes = []
    
    print("\nGraph nodes and labels:")
    for i, graph in enumerate(graph_list):
        nodes = list(graph.nodes())
        all_nodes.append(nodes)
        
        # Get the labels for this graph
        labels = nx.get_node_attributes(graph, 'label')
        
        # Print nodes with their labels
        print(f"Graph {i}: {nodes}")
        print(f"  Labels: ", end="")
        for node in nodes:
            label = labels.get(node, "No label")
            print(f"Node {node}:{label} ", end="")
        print()  # New line after each graph

        print("  Edges: ", graph.edges())

    print("\n")
