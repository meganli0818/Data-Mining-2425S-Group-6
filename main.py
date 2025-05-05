import networkx as nx
import matplotlib.pyplot as plt
import copy
from edge_apriori import apriori, print_graph_nodes_simple

def main():

    G = nx.Graph()

    try:
        with open("./mutag.txt") as f:
            graph_data = f.readlines()
        

        G = graph_reader(graph_data)
        print("done with processing graph")
        aprior = apriori(G, 0.3)

        print("Number of frequent subgraphs: ", len(aprior))


        
    except FileNotFoundError:
        print("Error: test.txt file not found")
    except Exception as e:
        print(f"An error occurred: {e}")

    

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

def test_apriori():
    graph1 = nx.Graph()
    graph1.add_edges_from([(1, 2), (2, 3)])

    graph2 = nx.Graph()
    graph2.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 4)])

    graph3 = nx.Graph()
    graph3.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 9)])

    singleton = nx.Graph().add_node(1)


if __name__ == "__main__":
    main()