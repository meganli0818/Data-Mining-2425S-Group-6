import networkx as nx
import numpy as np
from ullman_algo import UllmanAlgorithm

"""
Implementation of Edge-based Growth for Candidate Generation

"""

def edge_based_merge(G, P):
    """
    Generate candidate graphs by edge-based extension.
    
    For each edge in P, remove it, find a mapping of the reduced P into G,
    then reintroduce the edge (or add an extra connection) to form a candidate.
    
    Args:
        G: The larger graph
        P: The pattern graph to extend
        
    Returns:
        A list of candidate extended graphs (each with one extra edge) or None.
    """
    if len(P.edges()) >= len(G.edges()):
        return None
    
    merged_results = []
    for edge in list(P.edges()):
        # Create a copy of P with the edge removed.
        P_remove_edge = nx.Graph(P)
        P_remove_edge.remove_edge(*edge)
        
        # Run Ullman’s algorithm to map the reduced pattern into G.
        ullman = UllmanAlgorithm(G, P_remove_edge)
        if ullman.ullman():
            mapping = ullman.get_mapping() 
            # The removed edge endpoints in P.
            u, v = edge
            # Their corresponding nodes in G via the mapping.
            mapped_u = mapping[u]
            mapped_v = mapping[v]
            
            # 1: reintroduce the removed edge between the mapped endpoints.
            merged_graph = nx.Graph(G)
            merged_graph.add_edge(mapped_u, mapped_v)
            
            # 2: try an alternative by adding an extra edge.
            merged_graph2 = nx.Graph(merged_graph)
            for candidate in G.neighbors(mapped_u):
                if candidate != mapped_v and not merged_graph2.has_edge(mapped_v, candidate):
                    merged_graph2.add_edge(mapped_v, candidate)
                    break  
            
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
            new_candidates = edge_based_merge(freq_subgraphs_list[i], freq_subgraphs_list[j])
            if new_candidates is not None:
                for new_candidate in new_candidates:
                    # Check if the candidate is already generated
                    candidate_already_generated = False
                    for existing_candidate in candidates:
                        if nx.is_isomorphic(new_candidate, existing_candidate):
                            candidate_already_generated = True
                            break
                    # Add candidate if it is not already generated
                    if not candidate_already_generated:    
                        candidates.add(new_candidate)
    return candidates



# Check if all k-1 size subgraphs of a k size candidate are frequent.
def all_subgraphs_frequent(candidate, freq_subgraphs):
    # Check if all k-1 size subgraphs of the candidate are frequent
    for edge in candidate.edges():
        sub_of_candidate = nx.Graph(candidate)
        sub_of_candidate.remove_edge(*edge)
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
    freq_subgraphs = []
    singleton = nx.Graph()
    singleton.add_node(0)
    curr_freq_subgraphs = [singleton]
    for graph in curr_freq_subgraphs:
        print("Initial frequent subgraphs:", graph.edges())
    candidates = generate_candidates(curr_freq_subgraphs)
    for graph in candidates:
        print("Initial candidate:", graph.edges())
    k = 3

    while candidates and len(candidates) > 0:
        for new_freq_subgraph in curr_freq_subgraphs:
            freq_subgraphs.append(new_freq_subgraph)
        print("Current frequent subgraphs:", curr_freq_subgraphs)
        curr_freq_subgraphs = candidates
        print("Current candidates:", candidates)
        candidates = prune(generate_candidates(curr_freq_subgraphs), curr_freq_subgraphs)

        # Count support for each candidate
        candidate_supp = {}
        for graph in graph_dataset:
            for candidate in candidates:
                if candidate.number_of_nodes() <= graph.number_of_nodes():
                    ullman = UllmanAlgorithm(graph, candidate)
                    if ullman.ullman():
                        if candidate not in candidate_supp:
                            candidate_supp[candidate] = 1
                        else:
                            candidate_supp[candidate] += 1

        # Filter candidates by min_support
        candidates = {candidate: supp for candidate, supp in candidate_supp.items() if supp >= min_support}
        
        k += 1

    
    return freq_subgraphs
