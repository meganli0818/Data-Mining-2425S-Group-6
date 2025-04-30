import networkx as nx
from ullman_algo import UllmanAlgorithm
import math

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
    print("\n[DEBUG] Entering edge_based_merge")
    # Ensure the two graphs have the same edge count
    if G.number_of_edges() != P.number_of_edges():
        print(f"[DEBUG] Edge count mismatch: G={G.number_of_edges()}, P={P.number_of_edges()}")
        return []
    
    if G.number_of_edges() == 1 and P.number_of_edges() == 1:
        return k1_join(G, P)

    merged_results = []

    # Only consider the P-edge that does not already exist in G
    for u_p, v_p in P.edges():
        if G.has_edge(u_p, v_p):
            continue

        # We want u_p to be the anchor (degree > 1), 
        # and v_p to be the leaf (degree == 1)
        # This is for consistency, because later when we define a fresh node
        # we need to know which node (label) to refer
        if P.degree(u_p) < P.degree(v_p):
            u_p, v_p = v_p, u_p

        # Try removing this P-edge against each G-edge
        for u_g, v_g in G.edges():
            P_rem = nx.Graph(P)
            P_rem.remove_edge(u_p, v_p)
            G_rem = nx.Graph(G)
            G_rem.remove_edge(u_g, v_g)

            # Exact-match isomorphism on the k-1 graphs
            iso = UllmanAlgorithm(G_rem, P_rem)
            if not iso.ullman(exact_match=False):
                continue

            mapping = iso.get_mapping()
            print(f"[DEBUG] Found join: P-edge=({u_p},{v_p}) - G-edge=({u_g},{v_g}); mapping={mapping}")

            # Candidate 1: add back the join edge between mapped nodes (only if labels match)
           
            if G.degree(u_g) >= G.degree(v_g):
                g_leaf = v_g
            else:
                g_leaf = u_g

            # get the two labels
            p_leaf_label = P.nodes[v_p].get('label')
            g_leaf_label = G.nodes[g_leaf].get('label')

    
            if p_leaf_label == g_leaf_label:
                mu, mv = mapping[u_p], mapping[v_p]
                cand1 = nx.Graph(G)
                if not cand1.has_edge(mu, mv):
                    cand1.add_edge(mu, mv)
                    merged_results.append(cand1)
                    print(f"[DEBUG] Cand1 edges: {sorted(cand1.edges())}")

            # Candidate 2: attach the P-leaf (node) along with its edge itself to G
            leaf = v_p
            leaf_label = P.nodes[leaf]['label']

            cand2 = nx.Graph(G)
            # only add the leaf node if it's not already in G
            if leaf not in cand2.nodes():
                cand2.add_node(leaf, label=leaf_label)

            # hook it up to the mapped anchor
            cand2.add_edge(mapping[u_p], leaf)
            merged_results.append(cand2)
            print(f"[DEBUG] Cand2 edges: {sorted(cand2.edges())}")
            print(f"[DEBUG] Cand2 edges: {sorted(cand2.edges())}")

            # We only want the two candidates for the one differing edge

    print(f"[DEBUG] edge_based_merge generated {len(merged_results)} candidates\n")
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
    freq_list = list(freq_subgraphs)
    candidates = set()
    
    # Loop through all pairs of frequent subgraphs, merging each pair to create new candidates.
    for i in range(len(freq_list)):
        for j in range(i+1, len(freq_list)):
            G_i = freq_list[i]
            G_j = freq_list[j]

            print(f"\n[DEBUG] Pair (i={i}, j={j}):")
            print(f"  G_i edges: {list(G_i.edges(data=True))}")
            print(f"  G_j edges: {list(G_j.edges(data=True))}")

            new_candidates = edge_based_merge(G_i, G_j)

            if not new_candidates:
                print("  -> edge_based_merge returned 0 candidates")
            else:
                print(f"  -> edge_based_merge returned {len(new_candidates)} candidates:")
                for idx, cand in enumerate(new_candidates):
                    print(f"     Candidate {idx}: edges = {list(cand.edges(data=True))}")

            for new_candidate in new_candidates or []:
                is_dup = False
                for existing in candidates:
                    if UllmanAlgorithm(existing, new_candidate).ullman(exact_match=True):
                        is_dup = True
                        break
                if not is_dup:
                    candidates.add(new_candidate)

        print(f"\rGenerated with graph {i+1}/{len(freq_list)}...", end="")

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
    # handle debug flag
    local_debug = DEBUG if verbose is None else verbose
    original_debug = globals().get('DEBUG', False)
    globals()['DEBUG'] = local_debug

    min_support = math.ceil(min_freq * len(graph_dataset))
    freq_subgraphs = []

    # 1) Initial single-edge subgraphs
    single_edges = all_single_edge_graphs(graph_dataset)
    curr_freq = []
    for sub in single_edges:
        count = 0
        for G in graph_dataset:
            if sub.number_of_edges() <= G.number_of_edges():
                if UllmanAlgorithm(G, sub).ullman(exact_match=False):
                    count += 1
        if count >= min_support:
            curr_freq.append(sub)

    debug_print("Initial frequent single-edge count:", len(curr_freq))
    for sub in curr_freq:
        debug_print("  single-edge subgraph edges:", list(sub.edges()))
    print_graph_edges(curr_freq)

    # 2) Iteratively grow to k+1 edges
    round_k = 1
    while curr_freq:
        # Debug: show which curr_freq graphs will be added
        debug_print(f"Adding {len(curr_freq)} frequent subgraphs of size {round_k}:")
        for sub in curr_freq:
            debug_print("  edges:", list(sub.edges()))

        # commit k-edge subgraphs
        freq_subgraphs.extend(curr_freq)

        # generate (k+1)-edge candidates via edge-based merge
        unpruned = generate_candidates(curr_freq)
        print(f"generated candidates of size: {round_k + 1}")
        debug_print("raw candidates:")
        print_graph_edges(unpruned)

        # prune by checking all k-edge faces
        candidates = prune(unpruned, curr_freq)
        print(f"pruned candidates of size: {round_k + 1}")
        print("number of candidates:", len(candidates))

        # support count
        supp_map = {}
        for G in graph_dataset:
            for cand in candidates:
                if cand.number_of_edges() <= G.number_of_edges():
                    if UllmanAlgorithm(G, cand).ullman(exact_match=False):
                        supp_map[cand] = supp_map.get(cand, 0) + 1

        # filter by support
        curr_freq = [c for c, s in supp_map.items() if s >= min_support]
        round_k += 1
        print(f"number of candidates of size {round_k}:", len(curr_freq))
        for sub in curr_freq:
            debug_print(f"  next-round subgraph edges:", list(sub.edges()))
        print()

    # restore debug
    globals()['DEBUG'] = original_debug
    return freq_subgraphs



def print_graph_edges(graph_list, debug_only=True):
    if debug_only and not DEBUG:
        return
    for i, G in enumerate(graph_list):
        edges = [(G.nodes[u]['label'], G.nodes[v]['label']) for u, v in G.edges()]
        print(f"Graph {i}: Edges={edges}")
