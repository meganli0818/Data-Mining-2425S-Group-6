import networkx as nx
import matplotlib.pyplot as plt
import time
import copy
from ullman_algo.ullman_algo_node import UllmanAlgorithmNode
from ullman_algo.ullman_algo_edge import UllmanAlgorithmEdge
import Apriori_Node
import Apriori_Edge
import argparse
import sys

def main():
    """
    Main function to load graph data, run the Apriori algorithm, 
    and display execution statistics.
    """
    if len(sys.argv) == 2 and sys.argv[1] == '-h':
        print("Usage: python main.py <dataset file path> <algorithm type (edge or node)> <minimum frequency>")
        print("Example: python main.py ./test.txt edge 0.5")
        sys.exit(0)
    if len(sys.argv) != 4 or sys.argv[2] not in ['edge', 'node'] or float(sys.argv[3]) > 1:
        print("Usage: python main.py <dataset file path> <algorithm type (edge or node)> <minimum frequency>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    algorithm_type = sys.argv[2]
    min_frequency = float(sys.argv[3])

    try:
        # Read graph data from file (change file name to file that contains the dataset)
        with open(filepath) as f:
            graph_data = f.readlines()

        # Convert raw data into NetworkX graph objects
        graphs = graph_reader(graph_data)
        print("Finished with processing graphs\nBeginning Apriori algorithm...\n\n")

        if algorithm_type == 'edge':
            # Measure execution time of the Apriori algorithm
            start_time = time.time()
            apriori = Apriori_Edge.apriori(graphs, min_frequency)
            end_time = time.time()
        else:
            # Measure execution time of the Apriori algorithm
            start_time = time.time()
            apriori = Apriori_Node.apriori(graphs, min_frequency)
            end_time = time.time()
        
        print(f"Frequent subgraphs count: {len(apriori)}")
        print(f"Execution time: {end_time - start_time} seconds")

    except FileNotFoundError:
        print(f"Error: {filepath} not found")
    
    
    
    
    

def graph_reader(graph_data):
    """
    Parse a text file containing graph data in a specific format
    and convert it to NetworkX graph objects.
    
    Args:
        graph_data (list): Lines from the graph data file
        
    Returns:
        list: List of NetworkX graph objects
    """
    graphs = []
    G = nx.Graph()
    for line in graph_data:
        if line.startswith("t"):
            # 't' indicates a new graph, so add the current graph to the list
            # and start a new one (skip the first empty graph)
            graphs.append(G)
            G = nx.Graph()
        if line.startswith("v"):
            # 'v' lines define vertices with labels: v [node_id] [label]
            _, n, m = line.split()
            G.add_node(int(n), label=m)
        elif line.startswith("e"):
            # 'e' lines define edges: e [source] [target] [label]
            _, u, v, m2 = line.split()
            G.add_edge(int(u), int(v))
    return graphs[1:]  # Skip the first empty graph

if __name__ == "__main__":
    main()