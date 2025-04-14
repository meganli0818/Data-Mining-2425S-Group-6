import networkx as nx
from ullman_algo.ullman_algo import UllmanAlgorithm
import math

# Debug flag to control output verbosity
DEBUG = False  # Set to False for production mode

def debug_print(*args, **kwargs):
    """Print only if DEBUG is True."""
    if DEBUG:
        print(*args, **kwargs)

# Merges two graphs
def node_based_merge(G, P):
    if len(P.nodes()) != len(G.nodes()):
        return None
    merged_results = []
    for node in P.nodes():
        P_remove_node = nx.Graph(P)
        P_remove_node.remove_node(node)
        ullman = UllmanAlgorithm(G, P_remove_node)
        if ullman.ullman(False):
            unmapped_nodes = ullman.get_unmapped_vertices()
            G_remove_node = nx.Graph(G)
            for unmapped_node in unmapped_nodes:
                G_remove_node.remove_node(unmapped_node)  
            
            exact_match = UllmanAlgorithm(G_remove_node, P_remove_node)
            if exact_match.ullman(True):
                mapping = ullman.get_mapping()

                # Create a new graph by merging G and P
                merged_graph = nx.Graph(G)
                removed_node_neighbors = list(P.neighbors(node))
                
                new_node = max(G.nodes()) + 1 if G.nodes() else 1
                merged_graph.add_node(new_node, label=P.nodes[node]['label'])
            

                for neighbor in removed_node_neighbors:
                    merged_graph.add_edge(new_node, mapping[neighbor])
                # 1: connect the node we removed from P to the nodes in G corresponding to the isomorphism
                # 2: return two graphs, one where the vertex is connected to the unmapped vertex, and another where its not
                merged_graph2 = nx.Graph(merged_graph)
                for unmapped_node in unmapped_nodes:
                    merged_graph2.add_edge(new_node, unmapped_node)
                merged_results.append(merged_graph)
                merged_results.append(merged_graph2)
    return merged_results

# Generates candidate subgraphs of size k+1 from a frequent subgraph of size k.
def generate_candidates(freq_subgraphs):
    if freq_subgraphs is None or len(freq_subgraphs) == 0:
        return None
    freq_subgraphs_list = list(freq_subgraphs)
    candidates = set()
    # Loop through all pairs of frequent subgraphs, merging them to create new candidates
    for i in range(len(freq_subgraphs_list)):
        for j in range(i, len(freq_subgraphs_list)):
            new_candidates = node_based_merge(freq_subgraphs_list[i], freq_subgraphs_list[j])
            if new_candidates is not None:
                for new_candidate in new_candidates:
                    # Check if the candidate is already generated
                    candidate_already_generated = False
                    for existing_candidate in candidates:
                        ullman_exact = UllmanAlgorithm(existing_candidate, new_candidate)
                        # if the candidate is already in the list, no need to add it
                        if ullman_exact.ullman(True):
                            candidate_already_generated = True
                            break
                    # Add candidate if it is not already generated
                    if not candidate_already_generated:    
                        candidates.add(new_candidate)
                        #debug_print("candidate found")
        print(f"\rGenerated with graph {i}/{len(freq_subgraphs_list)}...", end="")

    print()
    return candidates

# Check if all k-1 size subgraphs of a k size candidate are frequent.
def all_subgraphs_frequent(candidate, freq_subgraphs):
    # Check if all k-1 size subgraphs of the candidate are frequent
    for node in candidate.nodes():
        sub_of_candidate = nx.Graph(candidate)
        sub_of_candidate.remove_node(node)
        sub_of_candidate_frequent = False
        for subgraph in freq_subgraphs:
            ullman = UllmanAlgorithm(subgraph, sub_of_candidate)
            if ullman.ullman(True):
                sub_of_candidate_frequent = True
                break
        if not sub_of_candidate_frequent:
            return False
    return True

# Prune candidates based on whether all subgraphs are frequent.
def prune(candidates, freq_subgraphs):
    pruned_candidates = set()
    for candidate in candidates:
        if all_subgraphs_frequent(candidate, freq_subgraphs):
            pruned_candidates.add(candidate)
    return pruned_candidates

def all_singletons(graph_dataset):
    unique_labels = set()
    singletons = []
    
    for graph in graph_dataset:
        # Get all labels for the current graph
        labels = nx.get_node_attributes(graph, 'label').values()
        
        # Add them to the set of unique labels
        unique_labels.update(labels)
    
    debug_print("labels found: ", unique_labels)
    
    for label in unique_labels:
        # Create a singleton graph for each unique label
        singleton_graph = nx.Graph()
        singleton_graph.add_node(0, label=label)
        
        # Add the singleton graph to the set of singletons
        singletons.append(singleton_graph)

    return singletons


# Apriori algorithm to find frequent subgraphs in a dataset of graphs.
def apriori(graph_dataset, min_freq, verbose=None):
    """
    Args:
        graph_dataset: List of graphs to mine
        min_freq: Minimum frequency threshold (0.0-1.0)
        verbose: Override global DEBUG setting (True/False/None)
    """
    # Use provided verbosity or fall back to global setting
    local_debug = DEBUG if verbose is None else verbose
    
    # Save original DEBUG value
    original_debug = globals()['DEBUG']
    globals()['DEBUG'] = local_debug
    
    min_support = math.ceil(min_freq * len(graph_dataset))
    freq_subgraphs = []

    singletons = all_singletons(graph_dataset)
    curr_freq_subgraphs = []
    for singleton in singletons:
        # Count support for each singleton
        candidate_supp = {}
        for graph in graph_dataset:
            if singleton.number_of_nodes() <= graph.number_of_nodes():
                ullman = UllmanAlgorithm(graph, singleton)
                if ullman.ullman(False):
                    if singleton not in candidate_supp:
                        candidate_supp[singleton] = 1
                    else:
                        candidate_supp[singleton] += 1
        
        for candidate, supp in candidate_supp.items():
            if supp >= min_support:
                curr_freq_subgraphs.append(candidate)
    
    debug_print("number of frequent singletons: ", len(curr_freq_subgraphs))
    debug_print("frequent singletons ")
    print_graph_nodes_simple(curr_freq_subgraphs)

    while curr_freq_subgraphs and len(curr_freq_subgraphs) > 0:
        freq_subgraphs.extend(curr_freq_subgraphs)

        unpruned_candidates = generate_candidates(curr_freq_subgraphs)
        print("generated candidates of size:", curr_freq_subgraphs[0].number_of_nodes() + 1)
        debug_print("generated candidates: ")
        print_graph_nodes_simple(unpruned_candidates)
        candidates = prune(unpruned_candidates, curr_freq_subgraphs)
        print("pruned candidates of size:", curr_freq_subgraphs[0].number_of_nodes() + 1)
        print("number of candidates: ", len(candidates))

        # Count support for each candidate
        candidate_supp = {}
        counter = 1
        for graph in graph_dataset:
            inner_counter = 1
            for candidate in candidates:
                if candidate.number_of_nodes() <= graph.number_of_nodes():
                    ullman = UllmanAlgorithm(graph, candidate)
                    if ullman.ullman(False):
                        if candidate not in candidate_supp:
                            candidate_supp[candidate] = 1
                        else:
                            candidate_supp[candidate] += 1
                
                print(f"\rChecked candidate {inner_counter}/{len(candidates)} with graph {counter}/{len(graph_dataset)}", end="")
                inner_counter += 1
            counter += 1
        
        print("\nCalculated support of size:", curr_freq_subgraphs[0].number_of_nodes() + 1)
        debug_print("number of potential candidates: ", len(candidate_supp))
        
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

        print("  Edges: ", len(graph.edges()))

    print("\n")