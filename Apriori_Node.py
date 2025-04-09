import networkx as nx

def node_based_merge(graph1, graph2):
    placeholder

def generate_candidates(freq_subgraphs, size):
    candidates = set()
    for i in range(len(freq_subgraphs)):
        for j in range(i, len(freq_subgraphs)):
            new_candidate = node_based_merge(freq_subgraphs[i], freq_subgraphs[j])
            if len(new_candidate.nodes()) == size:
                candidates.add(new_candidate)
    return candidates

# method to prune the candidates
def prune(candidates, freq_subgraphs):
    pruned_candidates = set()
    for candidate in candidates:
        all_subgraphs_freq = True
        for subgraph in freq_subgraphs:
            ## if not is_subgraph(candidate, subgraph):
                all_subgraphs_freq = False
                break
        if all_subgraphs_freq:
            pruned_candidates.add(candidate)
    return pruned_candidates

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
                if is_subgraph(graph, candidate):
                    if candidate not in candidate_supp:
                        candidate_supp[candidate] = 1
                    else    
                        candidate_supp[candidate] += 1

        # Filter candidates by min_support
        candidates = {candidate: supp for candidate, supp in candidate_supp.items() if supp >= min_support}
        
        k += 1

    return freq_subgraphs
