import networkx as nx
import matplotlib.pyplot as plt
import copy
from ullman_algo.ullman_algo import ullman

def main():
    G = nx.Graph()

    try:
        with open("./ullman_algo/smalltest.txt") as f:
            graph_data = f.readlines()
    
        for line in graph_data:
            if line.startswith("v"):
                _, n, m1 = line.split()
                G.add_node(int(n))
                print(n)
            elif line.startswith("e"):
                _, u, v, m2 = line.split()
                G.add_edge(int(u), int(v))

        P = copy.deepcopy(G)

        # remove an arbitrary edge from P to serve as the subgraph
        first_edge = next(iter(P.edges))
        P.remove_edge(*first_edge)
        print("done with processing graph")
    except FileNotFoundError:
        print("Error: test.txt file not found")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    ullman(G, P)

    plt.figure(1)
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Original Graph G")
    
    # Create second figure for graph P
    plt.figure(2)
    nx.draw(P, with_labels=True, node_color='lightpink', edge_color='gray')
    plt.title("Subgraph P")
    
    plt.show()
    

    


if __name__ == "__main__":
    main()