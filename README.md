Dynamic SSSP-Del Simulation
Fully Dynamic Single Source Shortest Path Visualizer

------------------------------------------------------------

1. OVERVIEW

This project provides an interactive simulation of the Dynamic Single Source Shortest Path (SSSP-Del) algorithm, implemented using a message-passing approach and visualized through a Streamlit-based dashboard.

The system updates shortest paths in real time when the graph changes, such as when edges or nodes are added, removed, or modified, without recomputing paths from scratch. This demonstrates the efficiency and practicality of dynamic graph algorithms in evolving networks.

------------------------------------------------------------

2. REAL-WORLD USE CASE: SOCIAL MEDIA SEARCH AND RECOMMENDATION

The project models a social media platform using graph-based representations.

Nodes represent users.
Edges represent social interactions such as follows, messages, likes, and comments.
Edge weights are derived from an engagement score.

Using Dynamic SSSP, the system supports:
- User search and ranking
- Recommendation of closely connected users
- Real-time adaptation to changing interactions

------------------------------------------------------------

3. WHY DYNAMIC SSSP

Traditional shortest path algorithms such as Dijkstra or Bellman-Ford assume static graphs and require full recomputation whenever the graph changes.

Dynamic SSSP-Del updates only affected regions of the graph, avoids full recomputation, supports real-time responsiveness, and is suitable for continuously changing systems.

------------------------------------------------------------

4. DATASET AND WEIGHT DESIGN

The application loads the graph from:
social_engagement.csv

Dataset fields:
user_id, username, friend_id, friend_name, location, friend_location,
likes, comments, messages, story_views, engagement_score, edge_weight

Synthetic data is used to preserve privacy and enable controlled experimentation.

------------------------------------------------------------

5. FEATURES

5.1 Dynamic SSSP-Del Algorithm
- Edge insertion, deletion, weight updates
- Node addition and removal
- Incremental shortest path updates

5.2 Asynchronous Message Passing
Event types:
DISTANCE_UPDATE, SET_TO_INFINITY, DISTANCE_QUERY

5.3 Interactive Streamlit Dashboard
Graph visualization, real-time updates, user search,
edge and node modification, batch CSV execution, benchmarking.

------------------------------------------------------------

6. BATCH UPDATE FEATURE

Supports sequential execution of multiple graph updates using batch_update.csv.
This extends the original research work.

------------------------------------------------------------

7. CLI VERSION

Command-line implementation available in finalsspdel.py.
Runnable in Jupyter Notebook or Google Colab.

------------------------------------------------------------

8. RESEARCH PAPER REFERENCE

Dynamic Single-Source Shortest Paths via Message Passing
arXiv: https://arxiv.org/pdf/2508.14319

------------------------------------------------------------

9. HOW TO RUN THE PROJECT

Install dependencies:
python -m pip install -r requirements.txt

Run:
streamlit run app.py

------------------------------------------------------------

10. SYSTEM REQUIREMENTS

Python 3.9 or later
Libraries: streamlit, pandas, networkx, matplotlib

------------------------------------------------------------

11. PROJECT STRUCTURE

Dynamic-SSSP-Simulation
app.py
finalsspdel.py
social_engagement.csv
dataset_generator.py
requirements.txt
README.txt

------------------------------------------------------------

12. SUMMARY

This project demonstrates Dynamic SSSP-Del for real-world search and recommendation systems in evolving networks.
