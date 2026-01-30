Dynamic SSSP-Del Simulation
Fully Dynamic Single Source Shortest Path Visualizer

This project provides an interactive simulation of Dynamic Single Source Shortest Path (SSSP) using a message-passing algorithm and a visual dashboard built with Streamlit.

The system updates shortest paths in real time when the graph changes—such as when edges or nodes are added, removed, or modified.
This project demontrate the use and representation of Dynamic SSSP-Del algorithm in real life scenario the project will show the representation in the middle and real life use is the seraching the recommended user.

"""
The Cli version is also there you can also run that by just simply running the python code in colab or jupyter (file name is finalsspdel.py).
"""

you can also modify the batch_update.csv according to the needs!!

HOW TO RUN THE PROJECT
 
""""""""""""""""""""""""""
1. Place required files:
   app.py
   social_engagement.csv
   requirements.txt

2. Install dependencies:
   pip install -r requirements.txt

3. Run application:
   streamlit run app.py

4. Open:
   http://localhost:8501/

"""""""""""""""""""""""""""
1. FEATURES

1. Dynamic SSSP Algorithm
   - Supports edge insertions
   - Supports edge deletions
   - Supports weight updates
   - Supports node addition & removal
   - Avoids full recomputation by updating only affected regions

2. Asynchronous Message Passing
   Uses event types such as:
   - DISTANCE_UPDATE
   - SET_TO_INFINITY
   - DISTANCE_QUERY

3. Interactive Streamlit Dashboard
   - Visual graph representation (NetworkX + Matplotlib)
   - Real-time shortest-path updates
   - Path highlight to any node
   - Search users by name
   - Add/Delete edges
   - Add/Delete nodes
   - Batch CSV execution //Novelty 
   - Performance benchmarking


2. DATASET FILE


The application loads the graph from:

social_engagement.csv
you can use the file provided or you create your own by running generate_data.py you can also change the number or nodes,name etc in generate_data.py
The dataset contains:
- user_id
- username
- friend_id
- friend_name
- location
- friend_location
- likes
- comments
- messages
- story_views
- engagement_score
- edge_weight

 
3. SYSTEM REQUIREMENTS
 

Python 3.9 or later

Libraries:
- streamlit
- pandas
- networkx
- matplotlib

4. PROJECT STRUCTURE
 Dynamic-SSSP-Simulation
│── main.py
│──finalsspdel.py
│── social_engagement.csv
│── dataset_generator.py  (optional)
│── requirements.txt
│── README.txt




Team Members

   Aman Singh (252IT002)
   Ankur Patel (252IT003)
   Deepasnhu (252IT006)
   Yash Kumbhawat (252IT033)