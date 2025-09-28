# Chinese Postman / Eulerian Trail Solver

This project is a Python implementation of an **Eulerian Trail / Chinese Postman Problem solver**.  
It can compute **Eulerian paths** or **Eulerian circuits** in graphs, and supports both **exact** and **heuristic** algorithms depending on the number of odd-degree vertices.

---

## ✨ Features
- Detects whether a graph has:
  - An **Eulerian cycle** (all vertices have even degree),
  - An **Eulerian trail** (exactly two vertices have odd degree),
  - Or requires edge duplication (Chinese Postman Problem).
- **Exact solver** (Dynamic Programming with bitmask) for small number of odd vertices.  
- **Heuristic solver** (k-NN sparse graph + Blossom or greedy matching) for large instances.  
- Integration with **NetworkX** for optimal minimum-weight matching (Blossom algorithm).  
- Route export in **JSON** or **CSV** format.  
- Optional **graph visualization** with Matplotlib.  
- Logging and timing information for each step.

---

## 🛠️ Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Farouc/chinese_postman.git
cd chinese_postman

# Create virtual environment
python -m venv venv
# Activate (Linux/Mac)
source venv/bin/activate
# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```


## 📦 Requirements

Minimal dependencies are:

```shell
networkx>=3.0
tqdm>=4.60
matplotlib>=3.4
```

All other imports (argparse, json, logging, etc.) are part of the Python standard library.


## 🚀 Usage

Basic usage:

```bash
python main.py -i <input_file> [options]
```

Example:

```bash
python main.py -i instances/paris_map.txt --plot_graph
```

This will:

    - Parse the graph instance `paris_map.txt`,

    - Solve for an Eulerian trail/circuit,

    - Print a summary in the console,

    - Display the graph.

## ⚙️ Main Options

* `-i, --input` : Path to the instance file (**required**).

* `--mode {auto,open,closed}` : Solver mode (default: `auto`).
  * `auto` → choose exact if *k* small, heuristic otherwise.
  * `open` → Eulerian trail (start ≠ end).
  * `closed` → Eulerian cycle (start = end).

* `--fast` : Force fast (heuristic) solver.

* `--exact` : Force exact DP solver (if k small).

* `--plot_graph` : Display the graph.

* `--print-route` : Print the trail (vertices or edges) in console.

---

Run with:

```bash
python main.py -h
```
## 🧪 Example Run

Run the solver on the sample instance `paris_map.txt` with graph plotting enabled:

```bash
python main.py -i instances/paris_map.txt --plot_graph
```

Output:

```text
19:08:15 | INFO | Odd-degree vertices: k=7318
19:08:15 | INFO | Using CLOSED (Chinese Postman) scalable solver.
k-NN (m=16): 100%|█████████████████████████████████████| 7318/7318 [00:00<00:00, 14799.40it/s]
19:08:15 | INFO | Sparse odd-graph edges: ~69751
19:08:15 | INFO | Matched 3529 pairs on sparse odd-graph

=== Solution Summary ===
Resolution mode        : closed_sparse
Total vertices         : 11348
Total edges            : 17958
Odd-degree vertices (k): 7318
Matched pairs          : 3529
Duplicated edges       : 4564
Added length           : 5252.0
Final trail length     : 22523
Total time             : 1.05 s
========================
```




