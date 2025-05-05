import networkx as nx
import matplotlib.pyplot as plt
import time
import copy
from ullman_algo.ullman_algo_node import UllmanAlgorithmNode
from ullman_algo.ullman_algo_edge import UllmanAlgorithmEdge
import Apriori_Node
import Apriori_Edge

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
        print("Error: test.txt file not found")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    test_apriori(graphs)
    
def test_apriori(graphs):

    frequent_count = []
    time_count = []
    
    # Measure execution time of the Apriori algorithm
    for i in range(1, 10, 1):
        start_time = time.time()
        apriori = Apriori_Edge.apriori(graphs, i*0.1)  # Run with 80% support threshold
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

def single_graph_reader(graph_data):
    """
    Parse a text file containing a single graph data in the same format.
    
    Args:
        graph_data (list): Lines from the graph data file
        
    Returns:
        nx.Graph: A single NetworkX graph object
    """
    G = nx.Graph()
    for line in graph_data:
            if line.startswith("v"):
                # Add vertices with labels
                _, n, m = line.split()
                G.add_node(int(n), label=m)
                print(n)
            elif line.startswith("e"):
                # Add edges
                _, u, v, m2 = line.split()
                G.add_edge(int(u), int(v))
    return G


def display_graph_with_labels(G, title=None, color='lightblue'):
    """
    Display a graph with node labels.
    
    Args:
        G (nx.Graph): NetworkX graph object to display
        title (str, optional): Title for the plot
        color (str, optional): Color for the nodes, default is lightblue
    """
    # Create a figure
    plt.figure(figsize=(8, 6))
    
    # Get node positions (layout)
    pos = nx.spring_layout(G, seed=42)  # Seed for reproducible layout
    
    # Get node labels (use node ID if no label attribute)
    labels = {}
    for node in G.nodes():
        if 'label' in G.nodes[node]:
            labels[node] = f"{node}:{G.nodes[node]['label']}"
        else:
            labels[node] = str(node)
    
    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_color=color, 
            node_size=700, edge_color='gray')
    
    # Draw the labels with a slight offset
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    
    # Add title if provided
    if title:
        plt.title(title)
    
    # Show the plot
    plt.axis('off')
    plt.tight_layout()

if __name__ == "__main__":
    main()