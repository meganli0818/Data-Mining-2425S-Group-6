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

    try:
        # Read graph data from file (change file name to file that contains the dataset)
        with open("./test.txt") as f:
            graph_data = f.readlines()

        # Convert raw data into NetworkX graph objects
        graphs = graph_reader(graph_data)

        print("done with processing graphs")
    except FileNotFoundError:
        print(f"Error: ./text.txt not found")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    test_apriori(graphs)
    
def test_apriori(graphs):

    frequent_count = []
    time_count = []
    
    # Measure execution time of the Apriori algorithm
    for i in range(3, 4, 1):
        start_time = time.time()
        apriori = Apriori_Node.apriori(graphs, i*0.1)  # Run with 80% support threshold
        end_time = time.time()

        time_count.append(end_time - start_time)
        frequent_count.append(len(apriori))

         # Write time data to file
    with open("time_data.txt", "w") as time_file:
        for time_value in time_count:
            time_file.write(f"{time_value}\n")
    
    # Write graph count data to file
    with open("graph_data.txt", "w") as graph_file:
        for count in frequent_count:
            graph_file.write(f"{count}\n")
    
    

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

