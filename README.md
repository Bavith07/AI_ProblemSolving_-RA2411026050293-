# AI Problem Solving Assignment


This repository contains the implementation of two AI problem-solving assignments:
1. **Tic-Tac-Toe AI** (Minimax vs Alpha-Beta Pruning)
2. **Smart Navigation System** (Breadth-First Search vs Depth-First Search)

---

## 📁 Folder Structure

```
AI_ProblemSolving/
│
├── app.py                     # Problem 1: Tic-Tac-Toe Flask Backend
├── static/                    # Problem 1: CSS and JS files
├── templates/                 # Problem 1: HTML templates
│
├── navigation/                # Problem 2: Smart Navigation System
│   ├── nav_app.py             # Problem 2: Navigation Flask Backend
│   ├── static/                # Problem 2: CSS and JS files
│   └── templates/             # Problem 2: HTML templates
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🎮 Problem 1: Tic-Tac-Toe AI

### Problem Description
The goal is to implement an unbeatable Tic-Tac-Toe AI that can play against a human. The AI evaluates all possible future moves to determine the optimal strategy, ensuring it either wins or draws.

### Algorithms Used
1. **Standard Minimax**: An adversarial search algorithm that explores the entire game tree to find the optimal move.
2. **Alpha-Beta Pruning**: An optimized version of Minimax that significantly reduces the number of nodes evaluated by pruning branches that cannot possibly influence the final decision.

### Execution Steps
1. Navigate to the root directory.
2. Install dependencies: `pip install -r requirements.txt` (only Flask is needed).
3. Run the application: `python app.py`
4. Open your browser and navigate to `http://127.0.0.1:5000`.

### Sample Output
When playing against the AI:
- **Minimax Nodes Explored**: ~59,784 (on first move)
- **Alpha-Beta Nodes Explored**: ~626 (on first move)
- **Speedup**: Alpha-Beta explores over 98% fewer nodes while making the exact same optimal move. The UI provides a real-time table of "Nodes Saved".

---

## 🗺️ Problem 2: Smart Navigation System

### Problem Description
The objective is to implement a pathfinding tool that allows users to interactively build a graph (cities/nodes and roads/edges) and then find the path between a Start and Goal node using two different graph traversal algorithms.

### Algorithms Used
1. **Breadth-First Search (BFS)**: Explores the graph level-by-level. It is guaranteed to find the shortest path in an unweighted graph.
2. **Depth-First Search (DFS)**: Explores as far as possible along a branch before backtracking. It does not guarantee the shortest path but can sometimes be more memory-efficient.

### Execution Steps
1. Navigate to the `navigation` folder: `cd navigation`
2. Run the application: `python nav_app.py`
3. Open your browser and navigate to `http://127.0.0.1:5001`.

### Sample Output
Given a complex grid network from A to Z:
- **BFS**: Finds the shortest path (e.g., 4 nodes long), exploring outward radially.
- **DFS**: Finds a potentially much longer path (e.g., 12 nodes long), zig-zagging deep into the graph before reaching the target.
- The UI visually maps both paths with differing colors and provides an exact comparison table of nodes explored and path length.

---

*Note: This repository maintains meaningful commits documenting the development process of the frontend, backend, and algorithm implementations.*
