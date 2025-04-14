import networkx as nx
import matplotlib.pyplot as plt
import time
import copy
from ullman_algo.ullman_algo import UllmanAlgorithm
import Apriori_Node

def main():

    try:
        with open("./ullman_algo/test.txt") as f:
            graph_data = f.readlines()

        graphs = graph_reader(graph_data)


        print("done with processing graphs")
    except FileNotFoundError:
        print("Error: test.txt file not found")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    
    # for graph in graphs:
    #     display_graph_with_labels(graph, title="Graph", color="lightblue")
    
    start_time = time.time()
    apriori = Apriori_Node.apriori(graphs, 0.7)
    end_time = time.time()



    print("Time taken for apriori:", end_time - start_time)
    print("number of frequent subgraphs:", len(apriori))

    # for graph in apriori:
    #     display_graph_with_labels(graph, title="Candidate Graph", color="lightgreen")


    plt.show()
    
    

    # plt.figure(1)
    # nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
    # plt.title("Original Graph G")
    
    # # Create second figure for graph P
    # plt.figure(2)
    # nx.draw(P, with_labels=True, node_color='lightpink', edge_color='gray')
    # plt.title("Subgraph P")
    
    # plt.show()
    
def graph_reader(graph_data):
    graphs = []
    G = nx.Graph()
    for line in graph_data:
        if line.startswith("t"):
            graphs.append(G)
            G = nx.Graph()
        if line.startswith("v"):
            _, n, m = line.split()
            G.add_node(int(n), label=m)
        elif line.startswith("e"):
            _, u, v, m2 = line.split()
            G.add_edge(int(u), int(v))
    return graphs[1:]

def single_graph_reader(graph_data):
    G = nx.Graph()
    for line in graph_data:
            if line.startswith("v"):
                _, n, m = line.split()
                G.add_node(int(n), label=m)
                print(n)
            elif line.startswith("e"):
                _, u, v, m2 = line.split()
                G.add_edge(int(u), int(v))
    return G

def test_apriori():
    # graph1 = nx.Graph()
    # graph1.add_edges_from([(1, 2), (2, 3)])

    # graph2 = nx.Graph()
    # graph2.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 4)])

    # graph3 = nx.Graph()
    # graph3.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 9), (2, 9)])

    graph2 = nx.Graph()
    graph2.add_edges_from([(1, 2), (2,3), (3,1)])

    graph3 = nx.Graph()
    graph3.add_edges_from([(1, 2), (2,3)])

    graph4 = nx.Graph()
    graph4.add_edges_from([(1, 2), (2,3), (3,1), (1,4)])

    singleton = nx.Graph().add_node(1)

    #merge23 = Apriori_Node.node_based_merge(graph2, graph3)
    #candidates23 = Apriori_Node.generate_candidates([graph2, graph3])
    
    #pruned = Apriori_Node.prune(candidates23, [graph2, graph3])
    apriori = Apriori_Node.apriori([graph2, graph3, graph4], 0.6)
    # print("Apriori Graphs:")
    # for graph in apriori:
    #     print(graph.edges())
    
    plt.figure(1)
    nx.draw(graph2, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Graph 2")

    plt.figure(2)
    nx.draw(graph3, with_labels=True, node_color='lightpink', edge_color='gray')
    plt.title("Graph 3")

    plt.figure(3)
    nx.draw(graph4, with_labels=True, node_color='lightyellow', edge_color='gray')
    plt.title("Graph 1")

    for graph in apriori:
        plt.figure()
        nx.draw(graph, with_labels=True, node_color='lightgreen', edge_color='gray')
        plt.title("Candidate Graph")


    plt.show()

def display_graph_with_labels(G, title=None, color='lightblue'):
    """
    Display a graph with node labels.
    
    Args:
        G: NetworkX graph object
        title: Optional title for the plot
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