import streamlit as st
import pandas as pd
import math

from graph_manager import SSSPGraphUI
from config import PAGE_CONFIG, CUSTOM_CSS, COLOR_LEGEND_HTML, GITHUB_URL
from tour import show_tour


def update_source_callback():
    selected_id = st.session_state.source_selector_value
    if st.session_state.graph.source_user_id != selected_id:
        st.session_state.graph.set_source(selected_id)
        st.toast(f"Source updated to {selected_id}", icon='🟢')


st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Session State Init ---
if 'graph' not in st.session_state:
    st.session_state.graph = SSSPGraphUI()
    try:
        df = pd.read_csv("social_engagement.csv")
        st.session_state.graph.load_data(df)
    except FileNotFoundError:
        st.error("CRITICAL: 'social_engagement.csv' not found.")
        st.stop()

if 'benchmark' not in st.session_state:
    st.session_state.benchmark = []
if 'search_result_id' not in st.session_state:
    st.session_state.search_result_id = None
if 'tour_seen' not in st.session_state:
    st.session_state.tour_seen = False
if 'tour_step' not in st.session_state:
    st.session_state.tour_step = 0
if 'source_selector_value' not in st.session_state and st.session_state.graph.all_users:
    st.session_state.source_selector_value = next(iter(st.session_state.graph.all_users.keys()))

g = st.session_state.graph
user_ids = sorted(list(g.all_users.keys()))

# --- Auto-trigger Tour ---
if not st.session_state.tour_seen:
    show_tour()

# --- Title & Metrics ---
st.title("🔗 SSSP Network Simulator")

if user_ids and g.source_user_id is not None:
    total_edges = sum(len(u.out_edges) for u in g.all_users.values())
    reachable = [u for u in g.all_users.values() if u.distance != math.inf and u.id != g.source_user_id]
    avg_dist = sum(u.distance for u in reachable) / len(reachable) if reachable else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Users", len(g.all_users))
    m2.metric("Connections", total_edges)
    m3.metric("Source Node", g.all_users[g.source_user_id].username)
    m4.metric("Avg Distance", f"{avg_dist:.3f}")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")

    if user_ids:
        src_idx = 0
        if g.source_user_id in user_ids:
            src_idx = user_ids.index(g.source_user_id)

        st.selectbox("Start Node (Source)", user_ids, index=src_idx,
            key='source_selector_value', on_change=update_source_callback)

        if g.source_user_id != st.session_state.source_selector_value:
            g.set_source(st.session_state.source_selector_value)

        if st.button("🔄 Force Recalculation"):
            g.set_source(g.source_user_id)
            st.toast(f"Reset source {g.source_user_id}", icon='🔄')

        st.divider()

        tab_search, tab_add, tab_del, tab_nodes, tab_batch = st.tabs(
            ["🔍 Search", "Add", "Del Edge", "Nodes", "📁 Batch"]
        )

        with tab_search:
            st.markdown("**Find User**")
            search_text = st.text_input("Enter Name")
            if st.button("Search"):
                best_id, matches = g.find_user_by_name(search_text)
                if best_id:
                    st.session_state.search_result_id = best_id
                    st.success(f"Found: {g.all_users[best_id].username} (ID {best_id})")
                else:
                    st.session_state.search_result_id = None
                    st.error("No user found.")
            if st.session_state.search_result_id:
                if st.button("📍 Visualize Path"):
                    st.session_state.viz_target = st.session_state.search_result_id
                    st.rerun()

        with tab_add:
            u_add = st.selectbox("From", user_ids, key="u_add")
            v_add = st.selectbox("To", user_ids, key="v_add")
            w_add = st.slider("Weight", 0.01, 1.0, 0.5)
            if st.button("Update Edge"):
                if u_add == v_add:
                    st.error("No self-loops.")
                else:
                    t = g.update_edge(u_add, v_add, w_add)
                    st.session_state.benchmark.append({"Type": "Edge Add", "Time": t})
                    st.success("Updated!")
                    st.rerun()

        with tab_del:
            u_del = st.selectbox("From", user_ids, key="u_del")
            v_del = st.selectbox("To", user_ids, key="v_del")
            if st.button("Delete Edge"):
                if v_del in g.all_users[u_del].out_edges:
                    t = g.delete_edge(u_del, v_del)
                    st.session_state.benchmark.append({"Type": "Edge Del", "Time": t})
                    st.warning("Deleted!")
                    st.rerun()
                else:
                    st.error("No edge exists.")

        with tab_nodes:
            st.markdown("**New Node**")
            n_id = st.number_input("ID", value=max(user_ids) + 1 if user_ids else 100)
            n_name = st.text_input("Name", "User X")
            if st.button("Add Node"):
                if g.add_node(n_id, n_name, "Web"):
                    st.success("Added!")
                    st.rerun()
                else:
                    st.error("ID exists.")
            st.markdown("**Del Node**")
            del_n = st.selectbox("ID", user_ids, key="del_node_sel")
            if st.button("Delete Node"):
                t = g.delete_node(del_n)
                if t == -1.0:
                    st.error("Can't delete Source.")
                else:
                    st.session_state.benchmark.append({"Type": "Node Del", "Time": t})
                    st.warning("Deleted!")
                    st.rerun()

        with tab_batch:
            st.markdown("**Batch CSV**")
            st.info("Cols: `operation`, `u_id`, `v_id`, `weight`, `username`")
            st.caption("Ops: add_edge, delete_edge, add_node, delete_node")

            uploaded_file = st.file_uploader("Upload CSV", type="csv")
            if uploaded_file:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.dataframe(batch_df.head(3))
                    if st.button("Process Batch"):
                        t = g.simulate_batch_update(batch_df)
                        st.session_state.benchmark.append({"Type": "Batch", "Time": t})
                        st.success(f"Done in {t:.5f}s!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    if st.button("🗺️ Take a Tour"):
        st.session_state.tour_seen = False
        st.session_state.tour_step = 0
        st.rerun()

    with st.expander("About this project"):
        st.markdown(f"""
**SSSP Network Simulator** demonstrates the Dynamic SSSP-Del algorithm
applied to social network analysis.

**Algorithm:** Message-passing based shortest path computation with
incremental updates — no full recomputation when the graph changes.

**Tech Stack:** Python, Streamlit, NetworkX, Matplotlib

**Use Case:** Friend recommendations, social graph analysis,
network connectivity scoring.

[View on GitHub]({GITHUB_URL})

**Team:** Aman Singh, Ankur Patel, Deepanshu, Yash Kumbhawat
""")

# --- Main Content ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Network Graph")
    if user_ids:
        idx = 0
        if 'viz_target' in st.session_state and st.session_state.viz_target in user_ids:
            opts = [None] + user_ids
            if st.session_state.viz_target in opts:
                idx = opts.index(st.session_state.viz_target)
        target_hl = st.selectbox("Highlight Path to:", [None] + user_ids, index=idx)
        path_list = None
        if target_hl:
            path_list = g.get_path(target_hl)
            if path_list:
                st.success(f"Path: {' → '.join(map(str, path_list))} | Cost: {g.all_users[target_hl].distance:.4f}")
            else:
                st.warning("Node is unreachable.")

        with st.spinner("Rendering graph..."):
            st.pyplot(g.get_graph_figure(highlight_path=path_list))

        st.markdown(COLOR_LEGEND_HTML, unsafe_allow_html=True)

with col2:
    st.subheader("User States")
    data = [
        {
            "ID": k,
            "Name": v.username,
            "Dist": f"{v.distance:.2f}" if v.distance != math.inf else "INF",
            "Pred": v.predecessor,
        }
        for k, v in g.all_users.items()
    ]
    df_status = pd.DataFrame(data)
    if not df_status.empty:
        df_status['ID'] = pd.to_numeric(df_status['ID']).astype(int)
        st.dataframe(df_status.sort_values("ID"), hide_index=True, height=400)
    else:
        st.info("No user data available.")

    st.subheader("Benchmark")
    if st.session_state.benchmark:
        df_bench = pd.DataFrame(st.session_state.benchmark)
        st.dataframe(df_bench, hide_index=True)
        st.line_chart(df_bench["Time"])

# --- Recommendations Section ---
st.divider()
st.subheader("🤝 People You May Know")

if g.source_user_id is not None:
    recommendations = g.get_recommendations(top_n=5)
    if recommendations:
        cols = st.columns(min(len(recommendations), 5))
        for i, rec in enumerate(recommendations):
            with cols[i]:
                st.markdown(f"""
<div style="background: linear-gradient(135deg, #f5f7fa, #e4e9f2);
            padding: 16px; border-radius: 12px; text-align: center;
            border: 1px solid #dfe6e9; min-height: 160px;">
    <div style="font-size: 2rem;">👤</div>
    <div style="font-weight: 700; font-size: 1rem; margin: 4px 0; color: #2C3E50;">
        {rec['username']}
    </div>
    <div style="font-size: 0.8rem; color: #7F8C8D;">ID: {rec['id']}</div>
    <div style="margin-top: 8px; font-size: 0.85rem;">
        <span style="color: #2ECC71; font-weight: 600;">Strength: {rec['strength']}</span><br>
        <span style="color: #7F8C8D;">{rec['hops']} hops away</span>
    </div>
</div>
""", unsafe_allow_html=True)
                if st.button(f"View Path", key=f"rec_{rec['id']}"):
                    st.session_state.viz_target = rec['id']
                    st.rerun()
    else:
        st.info("No recommendations available — all reachable users are already direct connections, or no indirect paths exist.")
else:
    st.info("Select a source node to see recommendations.")
