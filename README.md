# SSSP Network Simulator — Social Graph Search & Recommendation System

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-FF4B4B?logo=streamlit&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Analysis-4C9A2A)
![License](https://img.shields.io/badge/License-MIT-green)

An interactive simulation of the **Dynamic Single Source Shortest Path (SSSP-Del)** algorithm applied to social network analysis. The system computes shortest paths in real time and updates them incrementally when the graph changes — no full recomputation needed.

## Features

- **Dynamic SSSP Algorithm** — Supports edge insertions, deletions, weight updates, and node additions/removals with incremental path updates
- **Asynchronous Message Passing** — Uses event-driven convergence (`DISTANCE_UPDATE`, `SET_TO_INFINITY`, `DISTANCE_QUERY`)
- **Interactive Graph Visualization** — Real-time network rendering with path highlighting using NetworkX + Matplotlib
- **User Search** — Find users by name and visualize shortest paths
- **People You May Know** — Friend recommendations based on shortest path distance (non-neighbor reachable users)
- **Batch Operations** — Upload CSV files to perform bulk graph modifications
- **Performance Benchmarking** — Track execution time for each operation
- **Guided Tour** — First-visit onboarding dialog that walks through all features

## Research Reference

Based on: *Dynamic Single-Source Shortest Paths via Message Passing*
[arXiv:2508.14319](https://arxiv.org/pdf/2508.14319)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py

# Open in browser
# http://localhost:8501
```

## Dataset

The app loads graph data from `social_engagement.csv`. You can use the provided file or generate a new one:

```bash
python generate_data.py
```

**Dataset columns:** `user_id`, `username`, `friend_id`, `friend_name`, `location`, `friend_location`, `likes`, `comments`, `messages`, `story_views`, `engagement_score`, `edge_weight`

## Project Structure

```
├── app.py                  # Streamlit UI layer
├── sssp_algorithm.py       # Core algorithm (MessageType, User classes)
├── graph_manager.py        # Graph operations & recommendations
├── tour.py                 # Guided tour dialog
├── config.py               # Colors, CSS, constants
├── generate_data.py        # Dataset generator
├── social_engagement.csv   # Sample social network dataset
├── batch_update.csv        # Example batch operations
├── finalsspdel.py          # CLI/Jupyter notebook version
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
└── README.md
```

## How It Works

1. **Graph Construction** — Load social network data as a weighted directed graph
2. **SSSP Computation** — Select a source user; the algorithm computes shortest paths to all other users via message passing
3. **Dynamic Updates** — Add/delete edges or nodes; the algorithm updates only affected paths using `SET_TO_INFINITY` cascades and `DISTANCE_QUERY` recomputation
4. **Recommendations** — Users reachable via indirect paths (not direct neighbors) are ranked by distance and suggested as "People You May Know"

## System Requirements

- Python 3.9+
- Libraries: `streamlit`, `pandas`, `networkx`, `matplotlib`

## CLI Version

A command-line / Jupyter Notebook version is also available:

```bash
python finalsspdel.py
```

This can also be run in Google Colab.

## Team

- Aman Singh (252IT002)
- Ankur Patel (252IT003)
- Deepanshu (252IT006)
- Yash Kumbhawat (252IT033)
