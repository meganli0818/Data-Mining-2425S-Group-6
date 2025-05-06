# Data-Mining-2425S-Group-6  
**Frequent Subgraph Mining Final Project**

Welcome to Group 6's project for Data Mining 2425S! This repository contains our implementation of **Frequent Subgraph Mining (FSM)** algorithms, including:

- **Ullman's Subgraph Isomorphism Algorithm** (Node-based and Edge-based)
- **Apriori by Node**
- **Apriori by Edge**

---

## Project Structure

- `ullman_algo/` – contains our implementation of Ullman's algorithm  
- `Apriori_Node.py` – our implementation of the Apriori (by node) FSM algorithm
- `Apriori_Edge.py` – our implementation of the Apriori (by edge) FSM algorithm
- `main.py` – entry point for running FSM on your dataset

---

## How to Run

1. Add your dataset file (in **Graph Transaction Format** (see below) ) to the main directory.
2. Open `main.py` and change **line 15** to point to your dataset's filename.
3. Run the program:

   ```bash
   python main.py
   ```

---

## Graph Transaction Format Example

In the following example, numbers represent vertex numbers and letters represent labels.

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

