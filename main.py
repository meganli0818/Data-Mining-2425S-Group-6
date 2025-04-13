import networkx as nx
import matplotlib.pyplot as plt
import copy
from ullman_algo.ullman_algo import UllmanAlgorithm
from Apriori_Node import node_based_merge

def main():
    test_apriori()

    G = nx.Graph()

    # try:
    #     with open("./ullman_algo/test.txt") as f:
    #         graph_data = f.readlines()
        
    #     with open("./ullman_algo/Ptest.txt") as f:
    #         P_data = f.readlines()
    #     G = graph_reader(graph_data)

    #     P = graph_reader(P_data)

    #     print("done with processing graph")
    # except FileNotFoundError:
    #     print("Error: test.txt file not found")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    
    # try:
    #     matcher = UllmanAlgorithm(G, P)
    # except ValueError as e:
    #     print(f"Error: {e}")
    # if (matcher.ullman()):
    #     print("P is a subgraph of G")
    # else:
    #     print("P is NOT a subgraph of G")

    # print(matcher.get_unmapped_vertices())
    

    # plt.figure(1)
    # nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
    # plt.title("Original Graph G")
    
    # # Create second figure for graph P
    # plt.figure(2)
    # nx.draw(P, with_labels=True, node_color='lightpink', edge_color='gray')
    # plt.title("Subgraph P")
    
    # plt.show()
    

def graph_reader(graph_data):
    G = nx.Graph()
    for line in graph_data:
            if line.startswith("v"):
                _, n, m1 = line.split()
                G.add_node(int(n))
                print(n)
            elif line.startswith("e"):
                _, u, v, m2 = line.split()
                G.add_edge(int(u), int(v))
    return G

def test_apriori():
    graph1 = nx.Graph()
    graph1.add_edges_from([(1, 2), (2, 3)])

    graph2 = nx.Graph()
    graph2.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 4)])

    graph3 = nx.Graph()
    graph3.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 9)])

    merge23 = node_based_merge(graph2, graph3)
    print("Merged Graphs:")
    for merged_graph in merge23:
        print(merged_graph.edges())
    
    plt.figure(1)
    nx.draw(graph2, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Graph 2")

    plt.figure(2)
    nx.draw(graph3, with_labels=True, node_color='lightpink', edge_color='gray')
    plt.title("Graph 3")

    for graph in merge23:
        plt.figure()
        nx.draw(graph, with_labels=True, node_color='lightgreen', edge_color='gray')
        plt.title("Merged Graph")


    plt.show()

if __name__ == "__main__":
    main()