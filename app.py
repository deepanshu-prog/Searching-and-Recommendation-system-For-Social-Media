import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import time
from collections import deque, namedtuple
from enum import Enum
# 1. CORE ALGORITHM CLASSES

FLOAT_TOLERANCE = 1e-9

class MessageType(Enum):
    DISTANCE_UPDATE = "DistanceUpdate"
    SET_TO_INFINITY = "SetToInfinity"
    DISTANCE_QUERY = "DistanceQuery"
    ADD_TO_SUCCESSOR = "AddToSuccessor"
    REMOVE_FROM_SUCCESSOR = "RemoveFromSuccessor"

Message = namedtuple('Message', ['from_id', 'to_id', 'type', 'body'])

class User:
    def __init__(self, user_id, username, location):
        self.id = user_id
        self.username = username
        self.location = location
        self.distance = math.inf
        self.predecessor = None
        self.successors = set()
        self.marked_as_infinity = False
        self.out_edges = {} 
        self.in_edges = {}  

    def reset_sssp(self):
        self.distance = math.inf
        self.predecessor = None
        self.successors = set()
        self.marked_as_infinity = False

    def on_message(self, msg, message_queue, all_users):
        if msg.type == MessageType.DISTANCE_UPDATE:
            new_distance = msg.body['distance']
            if new_distance < (self.distance - FLOAT_TOLERANCE):
                if self.predecessor and self.predecessor in all_users:
                    message_queue.append(Message(self.id, self.predecessor, MessageType.REMOVE_FROM_SUCCESSOR, {}))
                
                self.distance = new_distance
                self.predecessor = msg.from_id
                
                if self.predecessor in all_users:
                    message_queue.append(Message(self.id, self.predecessor, MessageType.ADD_TO_SUCCESSOR, {}))

                for friend_id, weight in self.out_edges.items():
                    if friend_id != self.predecessor and friend_id in all_users:
                        message_queue.append(Message(self.id, friend_id, MessageType.DISTANCE_UPDATE, {'distance': self.distance + weight}))

        elif msg.type == MessageType.SET_TO_INFINITY:
            if self.marked_as_infinity or self.distance == math.inf : return
            self.distance = math.inf
            self.predecessor = None
            self.marked_as_infinity = True
            successors_to_notify = list(self.successors) 
            self.successors.clear() 
            for successor_id in successors_to_notify:
                 if successor_id in all_users:
                     successor_node = all_users.get(successor_id)
                     if successor_node and successor_node.predecessor == self.id:
                          message_queue.append(Message(self.id, successor_id, MessageType.SET_TO_INFINITY, {}))

        elif msg.type == MessageType.DISTANCE_QUERY:
            if self.distance != math.inf:
                weight_to_querier = self.out_edges.get(msg.from_id)
                if weight_to_querier is not None and msg.from_id in all_users:
                    message_queue.append(Message(self.id, msg.from_id, MessageType.DISTANCE_UPDATE, {'distance': self.distance + weight_to_querier}))

        elif msg.type == MessageType.ADD_TO_SUCCESSOR:
            if msg.from_id in all_users: self.successors.add(msg.from_id)

        elif msg.type == MessageType.REMOVE_FROM_SUCCESSOR:
            self.successors.discard(msg.from_id)


 
# 2. GRAPH MANAGER UI 
 

class SSSPGraphUI:
    def __init__(self):
        self.all_users = {} 
        self.message_queue = deque()
        self.source_user_id = None

    def load_data(self, df):
        for _, row in df.iterrows():
            uid, fid = int(row['user_id']), int(row['friend_id'])
            if uid not in self.all_users:
                self.all_users[uid] = User(uid, row['username'], row['location'])
            if fid not in self.all_users:
                fname = row['friend_name'] if pd.notna(row['friend_name']) else f"User {fid}"
                floc = row['friend_location'] if pd.notna(row['friend_location']) else "Unknown"
                self.all_users[fid] = User(fid, fname, floc)
            w = float(row['edge_weight'])
            self.all_users[uid].out_edges[fid] = w
            self.all_users[fid].in_edges[uid] = w 

    def set_source(self, source_id):
        self.source_user_id = source_id
        for u in self.all_users.values(): u.reset_sssp()
        src = self.all_users.get(source_id)
        if not src: return
        src.distance = 0
        self.message_queue.clear()
        for fid, w in src.out_edges.items():
            self.message_queue.append(Message(source_id, fid, MessageType.DISTANCE_UPDATE, {'distance': w}))
        self.process_queue()

    def process_queue(self):
        i = 0
        limit = 10000 
        while self.message_queue and i < limit:
            msg = self.message_queue.popleft()
            if msg.to_id in self.all_users:
                self.all_users[msg.to_id].on_message(msg, self.message_queue, self.all_users)
            i += 1

    def _trigger_recomputation(self):
        affected = [node for node in self.all_users.values() if node.marked_as_infinity]
        for node in self.all_users.values():
                if node.marked_as_infinity:
                    node.marked_as_infinity = False
                    for neighbor in node.in_edges:
                        self.message_queue.append(Message(node.id, neighbor, MessageType.DISTANCE_QUERY, {}))
        self.process_queue()

    # --- SINGLE OPERATIONS ---
    def update_edge(self, u, v, w):
        start = time.perf_counter()
        self.all_users[u].out_edges[v] = w
        self.all_users[v].in_edges[u] = w
        user_u = self.all_users[u]
        user_v = self.all_users[v]
        if user_u.distance != math.inf and (user_u.distance + w) < (user_v.distance - FLOAT_TOLERANCE):
            self.message_queue.append(Message(u, v, MessageType.DISTANCE_UPDATE, {'distance': user_u.distance + w}))
            self.process_queue()
        return time.perf_counter() - start

    def delete_edge(self, u, v):
        start = time.perf_counter()
        if v in self.all_users[u].out_edges: del self.all_users[u].out_edges[v]
        if u in self.all_users[v].in_edges: del self.all_users[v].in_edges[u]
        user_u = self.all_users[u]
        user_v = self.all_users[v]
        if user_v.predecessor == u:
            self.message_queue.append(Message(u, v, MessageType.SET_TO_INFINITY, {}))
            self.process_queue()
            self._trigger_recomputation()
        return time.perf_counter() - start

    def add_node(self, user_id, username, location):
        if user_id in self.all_users: return False
        self.all_users[user_id] = User(user_id, username, location)
        return True

    def delete_node(self, target_id):
        if target_id not in self.all_users or target_id == self.source_user_id: return -1.0
        start = time.perf_counter()
        target_user = self.all_users[target_id]
        for neighbor_id in list(target_user.in_edges.keys()):
            if neighbor_id in self.all_users:
                neighbor = self.all_users[neighbor_id]
                if target_id in neighbor.out_edges: del neighbor.out_edges[target_id]
        for neighbor_id in list(target_user.out_edges.keys()):
            if neighbor_id in self.all_users:
                neighbor = self.all_users[neighbor_id]
                if target_id in neighbor.in_edges: del neighbor.in_edges[target_id]
                if neighbor.predecessor == target_id:
                    self.message_queue.append(Message(target_id, neighbor_id, MessageType.SET_TO_INFINITY, {}))
        del self.all_users[target_id]
        self.process_queue()
        self._trigger_recomputation()
        return time.perf_counter() - start

    # --- BATCH PROCESSOR ---
    def simulate_batch_update(self, batch_df):
        
        if not self.source_user_id: return 0.0

        batch_df.columns = [c.lower().strip() for c in batch_df.columns]
        
        # Check if we have node operations
        has_node_ops = batch_df['operation'].str.contains('node', case=False).any()

        start_time = time.perf_counter()
        
        # MODE 1: SEQUENTIAL SAFE MODE 
        if has_node_ops:
            
            for _, row in batch_df.iterrows():
                op = row.get('operation', '').lower().strip()
                try:
                    uid = int(row.get('u_id'))
                except: continue

                if 'add_node' in op:
                    uname = str(row.get('username', f'User {uid}'))
                    uloc = str(row.get('location', 'Batch'))
                    self.add_node(uid, uname, uloc)
                
                elif 'delete_node' in op:
                    self.delete_node(uid)

                elif 'add_edge' in op or (op == 'add' and 'v_id' in row):
                    try:
                        vid = int(row.get('v_id'))
                        w = float(row.get('weight'))
                        if uid in self.all_users and vid in self.all_users:
                            self.update_edge(uid, vid, w)
                    except: continue

                elif 'delete_edge' in op or (op == 'delete' and 'v_id' in row):
                    try:
                        vid = int(row.get('v_id'))
                        if uid in self.all_users and vid in self.all_users:
                            self.delete_edge(uid, vid)
                    except: continue

      
        else:
            if self.message_queue: self.process_queue()
            initial_add_messages = []
            initial_delete_messages = []

            for _, row in batch_df.iterrows():
                try:
                    op = row.get('operation', '').lower().strip()
                    u_id = int(row.get('u_id'))
                    v_id = int(row.get('v_id'))
                except: continue

                u = self.all_users.get(u_id)
                v = self.all_users.get(v_id)
                if not u or not v: continue

                if 'add' in op:
                    try: w = float(row.get('weight'))
                    except: continue
                    u.out_edges[v_id] = w
                    v.in_edges[u_id] = w
                    if u.distance != math.inf and (u.distance + w) < (v.distance - FLOAT_TOLERANCE):
                        initial_add_messages.append(Message(u_id, v_id, MessageType.DISTANCE_UPDATE, {'distance': u.distance + w}))

                elif 'delete' in op:
                    old_w = u.out_edges.get(v_id, math.inf)
                    if v_id in u.out_edges: del u.out_edges[v_id]
                    if u_id in v.in_edges: del v.in_edges[u_id]
                    
                    is_critical = (v.predecessor == u_id) and (u.distance != math.inf)
                    if is_critical and old_w != math.inf and math.isclose(v.distance, u.distance + old_w, abs_tol=FLOAT_TOLERANCE):
                        initial_delete_messages.append(Message(u_id, v_id, MessageType.SET_TO_INFINITY, {}))

           
            for msg in initial_delete_messages: self.message_queue.append(msg)
            for msg in initial_add_messages: self.message_queue.append(msg)
            self.process_queue()
            self._trigger_recomputation()

        return time.perf_counter() - start_time


    def find_user_by_name(self, name_query):
        name_query = name_query.lower().strip()
        matches = []
        for u in self.all_users.values():
            if name_query in u.username.lower():
                matches.append(u)
        if not matches: return None, []
        matches.sort(key=lambda x: (x.distance, x.id))
        return matches[0].id, matches

    def get_graph_figure(self, highlight_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        G = nx.DiGraph()

        node_colors = []
        path_edges = set()
        path_nodes = set()
        dest_id = None
    
        if highlight_path:
            path_nodes = set(highlight_path)
            dest_id = highlight_path[-1]
            for i in range(len(highlight_path) - 1):
                path_edges.add((highlight_path[i], highlight_path[i + 1]))
    
        sorted_ids = sorted(self.all_users.keys())

        for uid in sorted_ids:
            user = self.all_users[uid]
            G.add_node(uid)
    
 
            if uid == self.source_user_id:
                color = '#00FF00'  # Source: Bright Green
            elif uid == dest_id:
                color = '#FF0000'  # Destination: Red
            elif uid in path_nodes:
                color = '#00FFFF'  # Intermediate: Cyan
            elif user.distance == math.inf:
                color = '#D3D3D3'  # Unreachable: Gray
            else:
                color = '#ADD8E6'  # Normal: Light Blue
    
            node_colors.append(color)
            for fid, w in user.out_edges.items():
                G.add_edge(uid, fid, weight=round(w, 2))

        if not G.nodes: return fig

        pos = nx.spring_layout(G, seed=42, k=1.5)

        ordered_nodes = sorted(G.nodes())
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=ordered_nodes,
            node_color=[node_colors[sorted_ids.index(n)] for n in ordered_nodes],
            node_size=800,
            edgecolors='black',
            ax=ax
        )

        labels = {}
        for n in G.nodes:
            user = self.all_users[n]
            if user.distance == math.inf:
                dist_str = "INF"
            elif n == self.source_user_id:
                dist_str = "0.00"
            else:
                dist_str = f"{user.distance:.2f}"
            labels[n] = f"{user.username}\n(ID: {n})\n{dist_str}"
    
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    
        standard_edges = [e for e in G.edges if e not in path_edges]
        nx.draw_networkx_edges(G, pos, edgelist=standard_edges, edge_color='gray', alpha=0.2, arrows=True, connectionstyle='arc3,rad=0.1', ax=ax)
        if path_edges:
            nx.draw_networkx_edges(G, pos, edgelist=list(path_edges), edge_color='#008B8B', width=2.5, arrows=True, connectionstyle='arc3,rad=0.1', ax=ax)
    
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=7, ax=ax)
    
        if self.source_user_id in self.all_users:
            src_name = self.all_users[self.source_user_id].username
            src_id = self.source_user_id
            ax.set_title(f"SSSP Tree | Source: {src_name} (ID {src_id})", fontsize=10)
    
        ax.axis('off')
        return fig


    def get_path(self, target_id):
        if target_id not in self.all_users or self.all_users[target_id].distance == math.inf:
            return None
        path = []
        curr = target_id
        limit = len(self.all_users) + 2
        while curr is not None and len(path) < limit:
            path.append(curr)
            if curr == self.source_user_id: break
            curr = self.all_users[curr].predecessor
        return path[::-1]


# 3. STREAMLIT UI LAYOUT

def update_source_callback_simple():
    selected_id = st.session_state.source_selector_value
    if st.session_state.graph.source_user_id != selected_id:
        st.session_state.graph.set_source(selected_id)
        st.toast(f"Source updated to {selected_id}", icon='🟢')
    

st.set_page_config(page_title="SSSP-Del Simulation", layout="wide")
st.title("🕸️ Dynamic SSSP Network Simulation")

if 'graph' not in st.session_state:
    st.session_state.graph = SSSPGraphUI()
    try:
        df = pd.read_csv("social_engagement.csv")
        st.session_state.graph.load_data(df)
    except FileNotFoundError:
        st.error("CRITICAL: 'social_engagement.csv' not found.")
        st.stop()

if 'benchmark' not in st.session_state: st.session_state.benchmark = []
if 'search_result_id' not in st.session_state: st.session_state.search_result_id = None
if 'source_selector_value' not in st.session_state and st.session_state.graph.all_users:
    st.session_state.source_selector_value = next(iter(st.session_state.graph.all_users.keys()))

g = st.session_state.graph
user_ids = sorted(list(g.all_users.keys()))

with st.sidebar:
    st.header("⚙️ Controls")
    
    if user_ids:
        src_idx = 0
        if g.source_user_id in user_ids: src_idx = user_ids.index(g.source_user_id)
        
        st.selectbox("Start Node (Source)", user_ids, index=src_idx,
            key='source_selector_value', on_change=update_source_callback_simple)
        
        if g.source_user_id != st.session_state.source_selector_value:
             g.set_source(st.session_state.source_selector_value)

        if st.button("🔄 Force Recalculation"):
            g.set_source(g.source_user_id)
            st.toast(f"Reset source {g.source_user_id}", icon='🔄')

        st.divider()
        
        tab_search, tab_add, tab_del, tab_nodes, tab_batch = st.tabs(["🔍 Search", "Add", "Del Edge", "Nodes", "📁 Batch"])

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
                if u_add == v_add: st.error("No self-loops.")
                else:
                    t = g.update_edge(u_add, v_add, w_add)
                    st.session_state.benchmark.append({"Type": "Edge Add", "Time": t})
                    st.success("Updated!")
                    time.sleep(0.5)
                    st.rerun()

        with tab_del:
            u_del = st.selectbox("From", user_ids, key="u_del")
            v_del = st.selectbox("To", user_ids, key="v_del")
            if st.button("Delete Edge"):
                if v_del in g.all_users[u_del].out_edges:
                    t = g.delete_edge(u_del, v_del)
                    st.session_state.benchmark.append({"Type": "Edge Del", "Time": t})
                    st.warning("Deleted!")
                    time.sleep(0.5)
                    st.rerun()
                else: st.error("No edge exists.")

        with tab_nodes:
            st.markdown("**New Node**")
            n_id = st.number_input("ID", value=max(user_ids)+1 if user_ids else 100)
            n_name = st.text_input("Name", "User X")
            if st.button("Add Node"):
                if g.add_node(n_id, n_name, "Web"):
                    st.success("Added!")
                    time.sleep(0.5)
                    st.rerun()
                else: st.error("ID exists.")
            st.markdown("**Del Node**")
            del_n = st.selectbox("ID", user_ids, key="del_node_sel")
            if st.button("Delete Node"):
                t = g.delete_node(del_n)
                if t == -1.0: st.error("Can't delete Source.")
                else:
                    st.session_state.benchmark.append({"Type": "Node Del", "Time": t})
                    st.warning("Deleted!")
                    time.sleep(0.5)
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
                        time.sleep(0.5)
                        st.rerun()
                except Exception as e: st.error(f"Error: {e}")

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
            if path_list: st.success(f"Path: {' → '.join(map(str, path_list))} | Cost: {g.all_users[target_hl].distance:.4f}")
            else: st.warning("Node is unreachable.")
        st.pyplot(g.get_graph_figure(highlight_path=path_list))

with col2:
    st.subheader("User States")
    data = [{"ID": k, "Name": v.username, "Dist": f"{v.distance:.2f}" if v.distance!=math.inf else "INF", "Pred": v.predecessor} for k,v in g.all_users.items()]
    df_status = pd.DataFrame(data)
    if not df_status.empty:
        df_status['ID'] = pd.to_numeric(df_status['ID']).astype(int) 
        st.dataframe(df_status.sort_values("ID"), hide_index=True, height=400)
    else: st.info("No user data available.")

    st.subheader("Benchmark")
    if st.session_state.benchmark:
        df_bench = pd.DataFrame(st.session_state.benchmark)
        st.dataframe(df_bench, hide_index=True)
        st.line_chart(df_bench["Time"])