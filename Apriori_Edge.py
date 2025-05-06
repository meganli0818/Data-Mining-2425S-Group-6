import networkx as nx
from ullman_algo.ullman_algo_edge import UllmanAlgorithmEdge
import math

# Debug flag to control output verbosity
DEBUG = False  # Set to False for production mode


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
    for u_p, v_p in (P.edges()):
        P_rem = nx.Graph(P)
        P_rem.remove_edge(u_p,v_p)
        iso = UllmanAlgorithmEdge(G, P_rem)

        # Check if the remaining "root" size k-1 P-graph is a subgraph of G.
        # If it is, we can merge the two graphs.

        if iso.ullman(False):
            unmapped_edges_g = iso.get_unmapped_edges_in_G()
            G_rem = nx.Graph(G)
            for unmapped_edge_g in (unmapped_edges_g):
                G_rem.remove_edge(*unmapped_edge_g)

            exact_match = UllmanAlgorithmEdge(G_rem, P_rem)
          
            if exact_match.ullman(True):
                unmapped_g_nodes = (iso.get_unmapped_vertices_in_G())
            
                mapping = iso.get_mapping()
                
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
                    # P doesn't have a unmapped node
                    if not merged_graph.has_edge(mapping[u_p], mapping[v_p]):
                        merged_graph.add_edge(mapping[u_p], mapping[v_p])
                        merged_results.append(merged_graph)

   
                    unmapped_node_g = next(iter(sorted(unmapped_g_nodes)))
                    if merged_graph.nodes[new_node]['label'] == G.nodes[unmapped_node_g]['label'] and not G.has_edge(unmapped_node_g, mapping[existing_node]): 
                        merged_graph2 = nx.Graph(G)
                        merged_graph2.add_edge(unmapped_node_g, mapping[existing_node])
                        merged_results.append(merged_graph2)
                        
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
    if len(shared_labels) < 1:
        return merged_results
    for label in shared_labels:
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
                    candidate_already_generated = False
                    for existing_candidate in candidates:
                        ullman_exact = UllmanAlgorithmEdge(existing_candidate, new_candidate)
                        
                        # No need to add candidate if it is already generated.
                        if ullman_exact.ullman(True):
                            candidate_already_generated = True
                            break

                    # Add candidate only if it is not already generated
                    if not candidate_already_generated and nx.is_connected(new_candidate):    
                        candidates.add(new_candidate)
        print(f"\rGenerated with graph {i+1}/{len(freq_subgraphs_list)}...", end="")

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


def all_single_edge_graphs(graph_dataset, frequent_singletons):
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
            # Sort labels to avoid duplicates (A,B) vs (B,A)
            label_u = graph.nodes[u].get('label')
            label_v = graph.nodes[v].get('label')
            if label_u is None or label_v is None:
                continue
            if label_u not in frequent_singletons or label_v not in frequent_singletons:
                continue
            pair = tuple(sorted((label_u, label_v)))
            unique_edge_labels.add(pair)

    debug_print("Unique edge label pairs:", unique_edge_labels)
    unique_edge_labels = sorted(unique_edge_labels)

    for label_u, label_v in unique_edge_labels:
        G = nx.Graph()
        G.add_node(0, label=label_u)
        G.add_node(1, label=label_v)
        G.add_edge(0, 1)
        single_edge_graphs.append(G)

    return single_edge_graphs

def all_singletons(graph_dataset):
    """
    Generates all singleton graphs from a dataset of graphs.

    This function extracts unique node labels from the input graph dataset 
    and creates singleton graphs (graphs with a single node) for each unique label.

    Args:
        graph_dataset (list): A list of NetworkX graph objects representing the dataset.

    Returns:
        list: A list of singleton graphs, where each graph contains a single node 
              with a unique label.

    Notes:
        - If a graph in the dataset does not have labeled nodes, those nodes will 
          not contribute to the singleton graphs.
    """
    unique_labels = set()
    singletons = []
    
    for graph in graph_dataset:
        # Get all labels for the current graph
        labels = nx.get_node_attributes(graph, 'label').values()
        
        # Add them to the set of unique labels
        unique_labels.update(labels)
    
    debug_print("labels found: ", unique_labels)

    unique_labels = sorted(unique_labels)
    
    for label in unique_labels:
        # Create a singleton graph for each unique label
        singleton_graph = nx.Graph()
        singleton_graph.add_node(0, label=label)
        
        # Add the singleton graph to the set of singletons
        singletons.append(singleton_graph)

    return singletons


def apriori(graph_dataset, min_freq):
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

    min_support = math.ceil(min_freq * len(graph_dataset))
    freq_subgraphs = []

    # Generate all singletons
    singletons = all_singletons(graph_dataset)
    curr_freq_subgraphs = []
    frequent_labels = []
    i = 1
    for singleton in singletons:
        candidate_supp = 0
        print(f"\rGenerating singleton {i+1}/{len(singletons)}...", end="")
        # Count support for each singleton
        for graph in graph_dataset:
            if candidate_supp >= min_support:
                curr_freq_subgraphs.append(singleton)
                frequent_labels.append(singleton.nodes[0]['label'])
                break
            if singleton.number_of_nodes() <= graph.number_of_nodes():
                    # Get the first node from singleton (which is always 0) and its label
                    first_node = next(iter(singleton.nodes()))
                    singleton_label = singleton.nodes[first_node]['label']
                    
                    # Check if any node in the graph has the same label
                    if any(graph.nodes[node].get('label') == singleton_label for node in graph.nodes()):
                        candidate_supp += 1

        i += 1
    
    print("Number of frequent singletons: ", len(curr_freq_subgraphs))
    print("\n\n")

    i=1
    single_edge_graphs = all_single_edge_graphs(graph_dataset, frequent_labels)
    for single_edge_graph in single_edge_graphs:
        print(f"\rGenerating single edge graph {i+1}/{len(single_edge_graphs)}...", end="")
        i += 1
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
                    if candidate_supp[single_edge_graph] >= min_support:
                        curr_freq_subgraphs.append(single_edge_graph)
                        break


    # Apriori algorithm
    i = 2
    while curr_freq_subgraphs and len(curr_freq_subgraphs) > 0:
        header = f" SIZE {i} "
        decoration = "=" * len(header)
        
        print("\n")
        print(decoration)
        print(header)
        print(decoration)
        print()
        # Generate candidates of size k+1 from current frequent subgraphs of size k
        freq_subgraphs.extend(curr_freq_subgraphs)
        unpruned_candidates = generate_candidates(curr_freq_subgraphs)
        print("\nFinished generating candidates of size", i)
        print("Number of unpruned candidates: ", len(unpruned_candidates))

        # Prune candidates
        candidates = prune(unpruned_candidates, curr_freq_subgraphs)
        print("\nFinished pruning candidates of size", i)
        print("Number of pruned candidates: ", len(candidates))
        print()

    


        # Count support for each candidate
        candidate_supp = {}
        curr_freq_subgraphs = []
        counter = 1
        for candidate in candidates:
            inner_counter = 1
            for graph in graph_dataset:
                if candidate.number_of_edges() <= graph.number_of_edges():
                    ullman = UllmanAlgorithmEdge(graph, candidate)
                    print(f"\rChecking candidate {counter}/{len(candidates)} with graph {inner_counter}/{len(graph_dataset)}    ", end="")
                    if ullman.ullman(False):
                        if candidate not in candidate_supp:
                            candidate_supp[candidate] = 1
                        else:
                            candidate_supp[candidate] += 1
                        if candidate_supp[candidate] >= min_support:
                            curr_freq_subgraphs.append(candidate)
                            break
                inner_counter += 1
            counter += 1
        
        print("\nFinished calculating support of size", i+1)
        print(f"Number of frequent subgraphs of size {i+1}: ", len(curr_freq_subgraphs))
        print("\n\n")
        i += 1
    

    return freq_subgraphs
