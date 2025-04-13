import networkx as nx
from ullman_algo.ullman_algo import UllmanAlgorithm

# Merges two graphs 
def node_based_merge(G, P):
    if len(P.nodes()) != len(G.nodes()):
        return None
    merged_results = []
    for node in P.nodes():
        P_remove_node = nx.Graph(P)
        P_remove_node.remove_node(node)
        ullman = UllmanAlgorithm(G, P_remove_node)
        if ullman.ullman():
            unmapped_nodes = ullman.get_unmapped_vertices()
            mapping = ullman.get_mapping()
            print("mapping", mapping)

            # Create a new graph by merging G and P
            merged_graph = nx.Graph(G)
            removed_node_neighbors = list(P.neighbors(node))
            
            new_node = max(G.nodes()) + 1 if G.nodes() else 1
            merged_graph.add_node(new_node)
            
            for node in merged_graph.nodes():
                print(node)

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
    for i in range(len(freq_subgraphs_list)):
        for j in range(i, len(freq_subgraphs_list)):
            new_candidates = node_based_merge(freq_subgraphs[i], freq_subgraphs[j])
            if new_candidates is not None:
                for candidate in new_candidates:
                    candidates.add(candidate)
    return candidates

# Check if all k-1 size subgraphs of a k size candidate are frequent.
def all_subgraphs_frequent(candidate, freq_subgraphs):
    for node in candidate.nodes():
        sub_of_candidate = candidate.remove_node(node)
        for subgraph in freq_subgraphs:
            ullman = UllmanAlgorithm(subgraph, sub_of_candidate)
            if not ullman.ullman():
                return False
    return True

# Prune candidates based on whether all subgraphs are frequent.
def prune(candidates, freq_subgraphs):
    pruned_candidates = set()
    for candidate in candidates:
        if all_subgraphs_frequent(candidate, freq_subgraphs):
            pruned_candidates.add(candidate)
    return pruned_candidates

# Apriori algorithm to find frequent subgraphs in a dataset of graphs.
def apriori(graph_dataset, min_freq):
    min_support = int(min_freq * len(graph_dataset))
    freq_subgraphs = set()
    singleton = nx.Graph()
    curr_freq_subgraphs = set().add(singleton)
    candidates = generate_candidates(curr_freq_subgraphs)
    k = 3

    while candidates and candidates.size() > 0:
        freq_subgraphs.add(curr_freq_subgraphs)
        curr_freq_subgraphs = candidates
        candidates = prune(generate_candidates(curr_freq_subgraphs), curr_freq_subgraphs)

        # Count support for each candidate
        candidate_supp = {}
        for graph in graph_dataset:
            for candidate in candidates:
                ullman = ullman_algo.UllmanAlgorithm(graph, candidate)
                if ullman.ullman():
                    if candidate not in candidate_supp:
                        candidate_supp[candidate] = 1
                    else:
                        candidate_supp[candidate] += 1

        # Filter candidates by min_support
        candidates = {candidate: supp for candidate, supp in candidate_supp.items() if supp >= min_support}
        
        k += 1

    
    return freq_subgraphs