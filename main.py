import networkx as nx
import matplotlib.pyplot as plt
import copy
from ullman_algo import UllmanAlgorithm
import node_apriori

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

    singleton = nx.Graph().add_node(1)

    merge23 = Apriori_Node.node_based_merge(graph2, graph3)
    candidates23 = Apriori_Node.generate_candidates(singleton)
    #pruned = Apriori_Node.prune(candidates23, [graph2, graph3])
    apriori = Apriori_Node.apriori([graph1, graph2, graph3], 0.5)
    print("Apriori Graphs:")
    for graph in apriori:
        print(graph.edges())
    
    plt.figure(1)
    nx.draw(graph2, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Graph 2")

    plt.figure(2)
    nx.draw(graph3, with_labels=True, node_color='lightpink', edge_color='gray')
    plt.title("Graph 3")

    plt.figure(3)
    nx.draw(graph1, with_labels=True, node_color='lightyellow', edge_color='gray')
    plt.title("Graph 1")

    for graph in apriori:
        plt.figure()
        nx.draw(graph, with_labels=True, node_color='lightgreen', edge_color='gray')
        plt.title("Candidate Graph")


    plt.show()

if __name__ == "__main__":
    main()