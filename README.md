# Data-Mining-2425S-Group-6  
**Frequent Subgraph Mining Final Project**

Welcome to Group 6's project for Data Mining 2425S! This repository contains our implementation of **Frequent Subgraph Mining (FSM)** algorithms, including:

- **Ullman's Subgraph Isomorphism Algorithm** (Node-based and Edge-based)
- **Apriori by Node**
- **Apriori by Edge**

---

## Project Structure

- `ullman_algo/` – contains both our Node-based and Edge-based implementations of Ullman's algorithm  
- `Apriori_Node.py` – our implementation of the Apriori (by node) FSM algorithm
- `Apriori_Edge.py` – our implementation of the Apriori (by edge) FSM algorithm
- `main.py` – entry point for running FSM on your dataset
- `testing material/` – contains code used for testing the algorithms. Feel free to take a look, but nothing important is in there.

---

## How to Run

1. Add your dataset file (in **Graph Transaction Format** (see below) ) to the main directory.
2. Run the program with the following usage:

   ```bash
   python main.py <dataset_file> <algorithm: edge or node> <support_threshold>
   ```

---

## Graph Transaction Format Example

In the following example, numbers represent vertex numbers and letters represent labels. Vertices must have labels, but edges can be unlabeled. Each graph in the dataset is separated by a line starting with `t #`, and each vertex is represented by `v <vertex_number> <label>`. Edges are represented by `e <source_vertex> <target_vertex> <label>`.

```
t # 0
v 0 A
v 1 B
v 2 C
e 0 1 X
e 1 2 Y
t # 1
v 0 A
v 1 C
v 2 B
e 0 1 Y
e 1 2 X
```

