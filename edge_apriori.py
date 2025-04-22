import networkx as nx
from ullman_algo import UllmanAlgorithm
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
    merged_results = []

    # 1. must have same # of edges
    k = G.number_of_edges()
    if P.number_of_edges() != k:
        return merged_results

    # 2. build sets of label‐edges
    edges_G = {(G.nodes[u]['label'], G.nodes[v]['label']) for u, v in G.edges()}
    edges_P = {(P.nodes[u]['label'], P.nodes[v]['label']) for u, v in P.edges()}

    # 3. require exactly k-1 in common
    common = edges_G & edges_P
    if len(common) != k - 1:
        return merged_results

    # 4. verify common (k-1)-edge subgraph is isomorphic
    def build_subgraph(edge_labels):
        H = nx.Graph()
        label2node = {}
        idx = 0
        for a, b in edge_labels:
            for lbl in (a, b):
                if lbl not in label2node:
                    label2node[lbl] = idx
                    H.add_node(idx, label=lbl)
                    idx += 1
            H.add_edge(label2node[a], label2node[b])
        return H

    common_G = build_subgraph(common)
    common_P = build_subgraph(common)
    if not UllmanAlgorithm(common_G, common_P).ullman(True):
        return merged_results

    # 5. build the merged (k+1)-edge graph
    merged_graph = nx.Graph()
    label2node = {}
    node_idx = 0

    for a, b in edges_G | edges_P:
        for lbl in (a, b):
            if lbl not in label2node:
                label2node[lbl] = node_idx
                merged_graph.add_node(node_idx, label=lbl)
                node_idx += 1
        merged_graph.add_edge(label2node[a], label2node[b])

    # 6. append to results and return
    merged_results.append(merged_graph)
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
                        ullman_exact = UllmanAlgorithm(existing_candidate, new_candidate)
                        
                        # No need to add candidate if it is already generated.
                        if ullman_exact.ullman(True):
                            candidate_already_generated = True
                            break

                    #  Add candidate only if it is not already generated
                    if not candidate_already_generated:    
                        candidates.add(new_candidate)
                        #debug_print("candidate found")
        print(f"\rGenerated with graph {i}/{len(freq_subgraphs_list)}...", end="")

    print()
    return candidates


def all_subgraphs_frequent(candidate, freq_subgraphs):
    """
    Checks if all (k-1)-size subgraphs of a k-size candidate graph are frequent.

    This function iteratively removes each node from the candidate graph to 
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
    for (u, v) in list(candidate.edges()):
        # 1. build the (k-1)-edge subgraph by removing edge (u,v)
        sub = nx.Graph(candidate)
        sub.remove_edge(u, v)

        # 2. only consider connected subgraphs
        if not nx.is_connected(sub):
            continue

        # 3. check if this sub is frequent
        found = False
        for freq in freq_subgraphs:
            if UllmanAlgorithm(freq, sub).ullman(True):
                found = True
                break

        # 4. if any connected subgraph isn't frequent, prune candidate
        if not found:
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


def all_singletons(graph_dataset):
    """
    Generates all singleton graphs (single-edge graphs) from a dataset of graphs.

    This function extracts unique edges (based on node labels) from the input graph dataset 
    and creates singleton graphs for each unique edge.

    Args:
        graph_dataset (list): A list of NetworkX graph objects representing the dataset.

    Returns:
        list: A list of singleton graphs, each containing exactly one edge connecting two labeled nodes.
    """
    unique_edges = set()
    singletons = []

    for graph in graph_dataset:
        for u, v in graph.edges():
            # Get sorted node labels to avoid duplicates like (A,B) vs (B,A)
            label_pair = tuple(sorted((graph.nodes[u]['label'], graph.nodes[v]['label'])))
            unique_edges.add(label_pair)

    debug_print("unique edges found: ", unique_edges)

    for label_u, label_v in unique_edges:
        singleton_graph = nx.Graph()
        singleton_graph.add_node(0, label=label_u)
        singleton_graph.add_node(1, label=label_v)
        singleton_graph.add_edge(0, 1)

        singletons.append(singleton_graph)

    return singletons

def apriori(graph_dataset, min_freq, verbose=None):
    """
    Apriori algorithm to find frequent subgraphs in a dataset of graphs.

    This function implements the Apriori algorithm to mine frequent subgraphs 
    from a dataset of graphs. It starts by identifying singleton graphs (graphs 
    with a single labeled node) and iteratively generates larger candidate 
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
    singletons = all_singletons(graph_dataset)
    curr_freq_subgraphs = []
    for singleton in singletons:
        # Count support for each singleton
        candidate_supp = {}
        for graph in graph_dataset:
            if singleton.number_of_edges() <= graph.number_of_edges():
                ullman = UllmanAlgorithm(graph, singleton)
                if ullman.ullman(False):
                    if singleton not in candidate_supp:
                        candidate_supp[singleton] = 1
                    else:
                        candidate_supp[singleton] += 1
        # Save singletons based on minimum support
        for candidate, supp in candidate_supp.items():
            if supp >= min_support:
                curr_freq_subgraphs.append(candidate)
    
    debug_print("number of frequent singletons: ", len(curr_freq_subgraphs))
    debug_print("frequent singletons ")
    print_graph_edges(curr_freq_subgraphs)

    # Apriori algorithm
    while curr_freq_subgraphs and len(curr_freq_subgraphs) > 0:
        
        # Generate candidates of size k+1 from current frequent subgraphs of size k
        freq_subgraphs.extend(curr_freq_subgraphs)
        unpruned_candidates = generate_candidates(curr_freq_subgraphs)
        print("generated candidates of size:", curr_freq_subgraphs[0].number_of_edges() + 1)
        debug_print("generated candidates: ")
        print_graph_edges(unpruned_candidates)

        # Prune candidates
        candidates = prune(unpruned_candidates, curr_freq_subgraphs)
        print("pruned candidates of size:", curr_freq_subgraphs[0].number_of_edges() + 1)
        print("number of candidates: ", len(candidates))

        # Count support for each candidate
        candidate_supp = {}
        counter = 1
        for graph in graph_dataset:
            inner_counter = 1
            for candidate in candidates:
                if candidate.number_of_edges() <= graph.number_of_edges():
                    ullman = UllmanAlgorithm(graph, candidate)
                    print(f"\rChecked candidate {inner_counter}/{len(candidates)} with graph {counter}/{len(graph_dataset)}    ", end="")
                    if ullman.ullman(False):
                        if candidate not in candidate_supp:
                            candidate_supp[candidate] = 1
                        else:
                            candidate_supp[candidate] += 1
                inner_counter += 1
            counter += 1
        
        print("\nCalculated support of size:", curr_freq_subgraphs[0].number_of_edges() + 1)
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


def print_graph_edges(graph_list, debug_only=True):
    if debug_only and not DEBUG:
        return
    for i, G in enumerate(graph_list):
        edges = [(G.nodes[u]['label'], G.nodes[v]['label']) for u, v in G.edges()]
        print(f"Graph {i}: Edges={edges}")
