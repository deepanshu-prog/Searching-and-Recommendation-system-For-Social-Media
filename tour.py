import streamlit as st

TOUR_STEPS = [
    {
        "title": "Welcome to SSSP Network Simulator",
        "icon": "👋",
        "content": """
This app simulates the **Dynamic Single Source Shortest Path (SSSP-Del)** algorithm on a social media network.

**What does it do?**
- Models a social network as a weighted graph where users are nodes and connections are edges
- Computes the shortest (strongest) path between any two users in real time
- Updates paths **incrementally** when the network changes — no full recomputation needed

This is the same type of algorithm that powers **friend suggestions** and **connection recommendations** on platforms like LinkedIn and Facebook.

Let's walk through the key features!
""",
    },
    {
        "title": "Step 1: Select a Source User",
        "icon": "🎯",
        "content": """
In the **sidebar**, you'll find the **Source Node** selector.

- Pick any user as the starting point — the algorithm computes shortest paths from this user to everyone else
- **Distance** represents connection strength — lower distance means a closer relationship
- A distance of **0** means it's the source user itself
- **INF** means the user is unreachable from the source

You can click **Force Recalculation** anytime to recompute all paths from scratch.
""",
    },
    {
        "title": "Step 2: Search & Visualize Paths",
        "icon": "🔍",
        "content": """
Use the **Search** tab in the sidebar to find users by name.

- Type a name (or part of it) and click **Search**
- Once found, click **Visualize Path** to highlight the shortest path on the graph

**Color Guide:**
- 🟢 **Green** — Source node (starting point)
- 🔴 **Red** — Destination node (who you searched for)
- 🔵 **Blue** — Intermediate nodes on the path
- ⚪ **Light Blue** — Other reachable nodes
- ⬜ **Gray** — Unreachable nodes
""",
    },
    {
        "title": "Step 3: Modify the Network",
        "icon": "✏️",
        "content": """
The real power of this algorithm is **dynamic updates** — you can change the network and watch paths update instantly.

**Add/Update Edges:**
- Go to the **Add** tab → select two users → set a weight → click **Update Edge**
- Weight ranges from 0.01 (very strong) to 1.0 (weak connection)

**Delete Edges:**
- Go to the **Del Edge** tab → select the edge to remove

**Add/Delete Nodes:**
- Go to the **Nodes** tab → add new users or remove existing ones
- The algorithm handles all cascading path updates automatically!
""",
    },
    {
        "title": "Step 4: Recommendations & Batch Ops",
        "icon": "🤝",
        "content": """
**People You May Know:**
- Below the graph, you'll see **recommended connections** — users who are close in the network but not directly connected to the source
- This is exactly how social media platforms suggest new friends!

**Batch Operations:**
- Use the **Batch** tab to upload a CSV file with multiple operations at once
- Supported operations: `add_edge`, `delete_edge`, `add_node`, `delete_node`

**Benchmark:**
- The performance panel tracks execution time for each operation
- See how the incremental algorithm avoids costly full recomputations!

---
You're all set! Close this dialog to start exploring. You can replay this tour anytime from the sidebar.
""",
    },
]


def render_tour():
    """Render tour inline using a container with overlay styling."""
    step = st.session_state.get("tour_step", 0)
    total = len(TOUR_STEPS)
    current = TOUR_STEPS[step]

    with st.container(border=True):
        st.progress((step + 1) / total, text=f"Step {step + 1} of {total}")
        st.markdown(f"### {current['icon']} {current['title']}")
        st.markdown(current["content"])

        cols = st.columns([1, 1, 1])

        with cols[0]:
            if step > 0:
                if st.button("← Previous", use_container_width=True, key="tour_prev"):
                    st.session_state.tour_step = step - 1
                    st.rerun()

        with cols[2]:
            if step < total - 1:
                if st.button("Next →", use_container_width=True, type="primary", key="tour_next"):
                    st.session_state.tour_step = step + 1
                    st.rerun()
            else:
                if st.button("Finish Tour ✓", use_container_width=True, type="primary", key="tour_finish"):
                    st.session_state.tour_active = False
                    st.session_state.tour_step = 0
                    st.rerun()
