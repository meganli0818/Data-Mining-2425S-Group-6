import networkx as nx
from ullman_algo.ullman_algo_node import UllmanAlgorithmNode
import math

# Debug flag to control output verbosity
DEBUG = False  # Set to False for production mode


def debug_print(*args, **kwargs):
    """Print only if DEBUG is True."""
    if DEBUG:
        print(*args, **kwargs)


# Merges two graphs
def node_based_merge(G, P):
    """
    Merges two graphs based on node-based subgraph isomorphism.

    This function attempts to merge two size k graphs, `G` and `P`, by iteratively 
    removing a node v from `P` to find a "root" graph of size k-1, and checking whether the 
    root is a subgraph of G using Ullman's algorithm. If a match is found, it  
    inserts v into G and connects v to all nodes in the root of G mapping to nodes that
    v was connected to in the root of P. Finally, it generates two size k+1 merged graphs: one where v
    is connected to the node of G not in the root, and another where v is not.

    Args:
        G (nx.Graph): The first graph to be merged.
        P (nx.Graph): The second graph to be merged.

    Returns:
        list: The two possible merged graphs. Each merged graph is a NetworkX 
              graph object. If no valid merges are found, an empty list 
              is returned.

    Notes:
        - If the number of nodes in `P` does not match the number of nodes 
          in `G`, the function returns `None`.
    """
    # Ensure P and G are the same size.
    if len(P.nodes()) != len(G.nodes()):
        return None
    merged_results = []
    
    # Loop through all nodes in P, removing one at a time.
    for node in P.nodes():
        P_remove_node = nx.Graph(P)
        P_remove_node.remove_node(node)
        ullman = UllmanAlgorithmNode(G, P_remove_node)

        # Check if the remaining "root" size k-1 graph is a subgraph of G.
        # If it is, we can merge the two graphs.
        if ullman.ullman(False):
            # Get the mapping of the nodes in root of P to root of G
            unmapped_nodes = ullman.get_unmapped_vertices()
            G_remove_node = nx.Graph(G)
            for unmapped_node in unmapped_nodes:
                G_remove_node.remove_node(unmapped_node)  
            exact_match = UllmanAlgorithmNode(G_remove_node, P_remove_node)
            if exact_match.ullman(True):
                mapping = ullman.get_mapping()

                # Create a new graph by merging G and P
                merged_graph = nx.Graph(G)
                removed_node_neighbors = list(P.neighbors(node))
                
                new_node = max(G.nodes()) + 1 if G.nodes() else 1
                merged_graph.add_node(new_node, label=P.nodes[node]['label'])
            
                # Connect the new node to the nodes in G that correspond to the isomorphism
                # between P and G roots
                for neighbor in removed_node_neighbors:
                    merged_graph.add_edge(new_node, mapping[neighbor])
                
                # Connect the new node to the node in G that is not in the root for
                # a second merged graph
                merged_graph2 = nx.Graph(merged_graph)
                for unmapped_node in unmapped_nodes:
                    merged_graph2.add_edge(new_node, unmapped_node)
                merged_results.append(merged_graph)
                merged_results.append(merged_graph2)
    return merged_results


def generate_candidates(freq_subgraphs):
    """
    Generates candidate subgraphs of size k+1 from a set of frequent subgraphs of size k.

    This function takes a list of frequent subgraphs of size k and generates 
    candidate subgraphs of size k+1 by node-based merging every pair of frequent subgraphs. 
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
            new_candidates = node_based_merge(freq_subgraphs_list[i], freq_subgraphs_list[j])
            if new_candidates is not None:
        
                # Check if each candidate is already generated.
                for new_candidate in new_candidates:
                    candidate_already_generated = False
                    for existing_candidate in candidates:
                        ullman_exact = UllmanAlgorithmNode(existing_candidate, new_candidate)
                        
                        # No need to add candidate if it is already generated.
                        if ullman_exact.ullman(True):
                            candidate_already_generated = True
                            break

                    #  Add candidate only if it is not already generated
                    if not candidate_already_generated and nx.is_connected(new_candidate):    
                        candidates.add(new_candidate)
        print(f"\rGenerated with graph: {i+1}/{len(freq_subgraphs_list)}...", end="")

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
    for node in candidate.nodes():
        sub_of_candidate = nx.Graph(candidate)
        sub_of_candidate.remove_node(node)
        if nx.is_connected(sub_of_candidate) is False:
            continue
        sub_of_candidate_frequent = False
        for subgraph in freq_subgraphs:
            ullman = UllmanAlgorithmNode(subgraph, sub_of_candidate)
            if ullman.ullman(True):
                sub_of_candidate_frequent = True
                break
        if not sub_of_candidate_frequent:
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
    
    for label in unique_labels:
        # Create a singleton graph for each unique label
        singleton_graph = nx.Graph()
        singleton_graph.add_node(0, label=label)
        
        # Add the singleton graph to the set of singletons
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
    i = 1
    
    for singleton in singletons:
        candidate_supp = 0
        print(f"\rGenerating singletons: {i}/{len(singletons)}...", end="")
        # Count support for each singleton
        for graph in graph_dataset:
            if candidate_supp >= min_support:
                curr_freq_subgraphs.append(singleton)
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
                if candidate.number_of_nodes() <= graph.number_of_nodes():
                    ullman = UllmanAlgorithmNode(graph, candidate)
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
        
        print("\nFinished calculating support of size", i)
        print(f"Number of frequent subgraphs of size {i}: ", len(curr_freq_subgraphs))
        print("\n\n")
        i += 1

    
    
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
