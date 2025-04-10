import networkx as nx
import numpy as np

'''
Edge-based Growth for Candidate Generation
'''

# In order for two graphs from Fk to be joined, a matching subgraph with (k − 1) edges needs
# to be present in the two graphs (graphs are joinable if they share exactly (k - 1) common edges)
def edge_based_join(graph1, graph2):
    edges1 = {tuple(sorted(edge)) for edge in graph1.edges()}
    edges2 = {tuple(sorted(edge)) for edge in graph2.edges()}

    if len(edges1) != len(edges2): # check for same # of edges // not checked during generate
        return None
    
    # both must have (k - 1) edges
    edge_diff = edges1.symmetric_difference(edges2) # unsure about this 
    if len(edge_diff) != 2:
        return None
    
    candidate_edges = edges1.union(edges2)
    if len(candidate_edges) != len(edges1) + 1:
        return None
    
    candidate = nx.Graph()
    candidate.add_edges_from(candidate_edges) # also unsure

    return candidate

def generate_candidates(freq_subgraphs):
    freq_subgraphs_list = list(freq_subgraphs)
    candidates = set()
    for i in range(len(freq_subgraphs_list)):
        for j in range(i, len(freq_subgraphs_list)):
            new_candidate = edge_based_join(freq_subgraphs[i], freq_subgraphs[j])
            if new_candidate:
                candidates.add(new_candidate)
    return candidates


# Check if all k-1 size subgraphs of a k size candidate are frequent.
def all_subgraphs_frequent(candidate, freq_subgraphs):
    for edge in candidate.edges():
        sub_of_candidate = candidate.remove_edge(*edge)
        for subgraph in freq_subgraphs:
            # if not is_subgraph(sub_of_candidate, subgraph):
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
def apriori(graph_dataset, min_support):
    freq_subgraphs = set()
    singleton = nx.Graph()
    curr_freq_subgraphs = set().add(singleton)
    candidates = generate_candidates(curr_freq_subgraphs, 2)
    k = 3

    while candidates.size() > 0:
        freq_subgraphs.add(curr_freq_subgraphs)
        curr_freq_subgraphs = candidates
        candidates = prune(generate_candidates(curr_freq_subgraphs, k), curr_freq_subgraphs)

        # Count support for each candidate
        candidate_supp = {}
        for graph in graph_dataset:
            for candidate in candidates:
                # if is_subgraph(graph, candidate):
                    if candidate not in candidate_supp:
                        candidate_supp[candidate] = 1
                    else:    
                        candidate_supp[candidate] += 1

        # Filter candidates by min_support
        candidates = {candidate: supp for candidate, supp in candidate_supp.items() if supp >= min_support}
        
        k += 1

    return freq_subgraphs