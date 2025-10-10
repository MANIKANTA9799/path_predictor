

import streamlit as st
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import warnings
warnings.filterwarnings("ignore")


DEFAULT_NUM_NODES = 24
DEFAULT_SEQ_LEN = 3
DEFAULT_EPOCHS = 10   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(layout="wide", page_title="Attack Path Predictor")



def generate_temporal_graphs(num_nodes=24, seq_len=3, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    graphs = []
    base = nx.erdos_renyi_graph(n=num_nodes, p=0.08, seed=seed)
    for i in range(num_nodes):
        if i not in base:
            base.add_node(i)
    types = {}
    for n in base.nodes():
        types[n] = random.choices(['user','workstation','server','router','firewall'],
                                  [0.45,0.30,0.15,0.07,0.03])[0]
    prev = base.copy()
    for t in range(seq_len):
        G = prev.copy()
        for _ in range(max(1, num_nodes//12)):
            a = random.randrange(num_nodes); b = random.randrange(num_nodes)
            if a==b: continue
            if random.random() < 0.5:
                G.add_edge(a,b)
            else:
                if G.has_edge(a,b) and random.random() < 0.3:
                    G.remove_edge(a,b)

        for n in G.nodes():
            ntype = types[n]
            vuln_base = {'user':0.05,'workstation':0.12,'server':0.20,'router':0.1,'firewall':0.02}[ntype]
            cvss = round(random.uniform(0,10),1) if random.random() < 0.25 else 0.0
            is_vuln = 1 if (cvss>0 or random.random() < vuln_base) else 0
            privilege = {'user':0,'workstation':0,'server':2,'router':1,'firewall':2}[ntype]
            G.nodes[n]['type'] = ntype
            G.nodes[n]['is_vulnerable'] = is_vuln
            G.nodes[n]['privilege'] = privilege
            G.nodes[n]['cvss'] = cvss
            G.nodes[n]['has_firewall'] = 1 if ntype=='firewall' else 0

        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps)-1):
                a = random.choice(list(comps[i])); b = random.choice(list(comps[i+1]))
                G.add_edge(a,b)
        graphs.append(G)
        prev = G
    return graphs




def label_attack_paths(graphs, num_attackers=1, num_targets=1, seed=None):
    if seed is not None:
        random.seed(seed)
    final = graphs[-1]
    nodes = list(final.nodes())
    attacker_pool = [n for n,d in graphs[0].nodes(data=True) if d['type'] in ('user','workstation')]
    if not attacker_pool:
        attacker_pool = nodes[:max(1,len(nodes)//4)]
    target_pool = [n for n,d in final.nodes(data=True) if d['type']=='server']
    if not target_pool:
        target_pool = nodes[-max(1,len(nodes)//8):]
    attackers = random.sample(attacker_pool, min(len(attacker_pool), num_attackers))
    targets = random.sample(target_pool, min(len(target_pool), num_targets))
    risky = {n:0 for n in nodes}
    for a in attackers:
        for t in targets:
            try:
                p = nx.shortest_path(final, a, t)
                for v in p:
                    risky[v] = 1
            except nx.NetworkXNoPath:
                pass
    for n,d in final.nodes(data=True):
        if d.get('is_vulnerable',0)==1 and random.random()<0.08:
            risky[n]=1
    return risky



def graphs_to_tensors(graphs):
   
    type_to_idx = {'user':0,'workstation':1,'server':2,'router':3,'firewall':4}
    N = graphs[0].number_of_nodes()
    xs = []
    edge_indices = []
    for G in graphs:
        feats = []
        for i in range(N):
            d = G.nodes[i]
            onehot = [0]*5
            onehot[type_to_idx[d['type']]] = 1
            feats.append(onehot + [d.get('is_vulnerable',0), d.get('privilege',0)/2.0, d.get('has_firewall',0), d.get('cvss',0.0)/10.0])
        x = torch.tensor(feats, dtype=torch.float).to(DEVICE)  
        xs.append(x)
        edges = []
        for u,v in G.edges():
            edges.append([u,v]); edges.append([v,u])
        if len(edges)==0:
            edge_idx = torch.zeros((2,0), dtype=torch.long).to(DEVICE)
        else:
            edge_idx = torch.tensor(edges, dtype=torch.long).t().contiguous().to(DEVICE)
        edge_indices.append(edge_idx)
    return xs, edge_indices



class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

class TemporalGNN(nn.Module):
    def __init__(self, feat_dim=9, gcn_hidden=64, lstm_hidden=64):
        super().__init__()
        self.encoder = GCNEncoder(feat_dim, gcn_hidden)
        self.lstm = nn.LSTM(input_size=gcn_hidden, hidden_size=lstm_hidden, num_layers=1, batch_first=False)
        self.head = nn.Linear(lstm_hidden, 1)
    def forward(self, xs_list, edge_index_list):
      
        embeds = []
        for x, ei in zip(xs_list, edge_index_list):
            if ei.numel()==0:
              
                lin = nn.Linear(x.shape[1], self.encoder.conv1.out_channels).to(x.device)
                emb = F.relu(lin(x))
            else:
                emb = self.encoder(x, ei)
            embeds.append(emb)  
        seq = torch.stack(embeds, dim=0)  
        lstm_out, _ = self.lstm(seq)       
        final = lstm_out[-1]              
        logits = self.head(final).view(-1) 
        return logits


# Loss, train for a single sample 

def train_single_sample(model, xs, eis, labels_tensor, epochs=10, lr=0.01):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    losses = []
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(xs, eis)
        loss = criterion(logits, labels_tensor.to(DEVICE))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses


# Mitigation recommender 

def recommend_mitigations(G_final, model_scores, top_k=5):
    deg = nx.degree_centrality(G_final)
    ranking = sorted(list(G_final.nodes()), key=lambda n: (model_scores.get(n,0.0)*(deg.get(n,0.0)+1.0)), reverse=True)
    attacker_pool = [n for n,d in G_final.nodes(data=True) if d['type'] in ('user','workstation')]
    target_pool = [n for n,d in G_final.nodes(data=True) if d['type']=='server']
    if not attacker_pool or not target_pool:
        return ranking[:top_k]
    a = random.choice(attacker_pool); t = random.choice(target_pool)
    selected = []
    Gtmp = G_final.copy()
    for node in ranking:
        if len(selected) >= top_k:
            break
        if node not in Gtmp: continue
        Gtmp.remove_node(node)
        try:
            _ = nx.shortest_path(Gtmp, a, t)
            still = True
        except:
            still = False
        if not still:
            selected.append(node)
    for node in ranking:
        if len(selected) >= top_k: break
        if node not in selected: selected.append(node)
    return selected


# Visualization helper

def plot_graph_with_scores(G, scores, true_labels=None, title="Graph"):
    pos = nx.spring_layout(G, seed=42)
    node_colors = []
    borders = []
    sizes = []
    for n in G.nodes():
        s = scores.get(n, 0.0)
        node_colors.append((s, 1.0 - s, 0.1))
        borders.append('blue' if true_labels and true_labels.get(n,0)==1 else 'black')
        sizes.append(100 + int(s*400))
    plt.figure(figsize=(6,5))
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors=borders, linewidths=1.2, node_size=sizes)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)


# Streamlit UI layout

st.title("Attack Path Predictor (PyTorch + PyG)")


col1, col2 = st.columns(2)

with col1:
    num_nodes = st.number_input("Number of nodes", min_value=8, max_value=120, value=DEFAULT_NUM_NODES, step=2)
    seq_len = st.number_input("Sequence length (timesteps)", min_value=2, max_value=6, value=DEFAULT_SEQ_LEN)
    epochs = st.number_input("Training epochs (quick demo)", min_value=2, max_value=200, value=DEFAULT_EPOCHS)
    seed_val = st.number_input("Random seed (optional)", value=42)

with col2:
    run_demo = st.button("Run quick demo (train tiny model)")
    upload = st.file_uploader("Or upload a JSON sequence of graphs (not required)", type=["json"])
    st.write("Note: JSON upload expects a list of graph snapshots; see README for format.")

st.write("---")

if run_demo:
    st.write("Generating synthetic temporal graph sequence...")
    graphs = generate_temporal_graphs(num_nodes=int(num_nodes), seq_len=int(seq_len), seed=int(seed_val))
    labels = label_attack_paths(graphs)
    st.write(f"Labels (nodes marked risky on final timestep): {sum(labels.values())}/{len(labels)}")

    st.write("Converting to tensors and training a small Temporal GNN...")
    xs, eis = graphs_to_tensors(graphs)
    # build labels tensor (final graph)
    y = torch.tensor([labels[i] for i in range(len(graphs[-1].nodes()))], dtype=torch.float).to(DEVICE)

    model = TemporalGNN(feat_dim=9, gcn_hidden=64, lstm_hidden=64).to(DEVICE)
    model, losses = train_single_sample(model, xs, eis, y, epochs=int(epochs), lr=0.01)
    st.write("Training finished. Loss trend (last values):", np.round(losses[-5:],4).tolist())

    st.write("Running inference on final graph...")
    model.eval()
    with torch.no_grad():
        logits = model(xs, eis)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
    model_scores = {i: float(probs[i]) for i in range(len(graphs[-1].nodes()))}
    st.subheader("Top risky nodes by model score")
    topk = sorted(model_scores.items(), key=lambda kv: kv[1], reverse=True)[:8]
    st.write(topk)

    st.subheader("Visualization (final timestep)")
    plot_graph_with_scores(graphs[-1], model_scores, true_labels=labels, title="Final graph: predicted risk (red=high)")

    st.subheader("Automated mitigation suggestions (top 5 nodes to patch):")
    recs = recommend_mitigations(graphs[-1], model_scores, top_k=5)
    st.write(recs)

if upload:
    import json
    st.write("Parsing uploaded JSON (not fully validated). Expecting list of snapshots; each snapshot: {nodes: [{id:int, attrs:{...}}], edges: [[u,v], ...]}")

    content = upload.read().decode('utf-8')
    try:
        data = json.loads(content)
        # convert to NetworkX graphs
        graphs = []
        for snap in data:
            G = nx.Graph()
            for nd in snap.get("nodes", []):
                nid = nd["id"]
                G.add_node(nid)
                for k,v in nd.get("attrs", {}).items():
                    G.nodes[nid][k]=v
            for e in snap.get("edges", []):
                G.add_edge(e[0], e[1])
            graphs.append(G)
        if len(graphs) < 2:
            st.error("Need at least 2 snapshots in the sequence.")
        else:
            st.success(f"Loaded {len(graphs)} snapshots. Running inference (no training)...")
            xs, eis = graphs_to_tensors(graphs)
           
            y_dummy = torch.zeros(len(graphs[-1].nodes()), dtype=torch.float).to(DEVICE)
            model = TemporalGNN(feat_dim=9, gcn_hidden=64, lstm_hidden=64).to(DEVICE)
            model, _ = train_single_sample(model, xs, eis, y_dummy, epochs=6, lr=0.01)  
            model.eval()
            with torch.no_grad():
                logits = model(xs, eis)
                probs = torch.sigmoid(logits).cpu().numpy()
            model_scores = {i: float(probs[i]) for i in range(len(graphs[-1].nodes()))}
            st.subheader("Predicted scores on uploaded graph :")
            st.write(sorted(model_scores.items(), key=lambda kv: kv[1], reverse=True)[:10])
            plot_graph_with_scores(graphs[-1], model_scores, title="Uploaded graph: predicted risk")
    except Exception as e:
        st.error("Failed to parse JSON upload: " + str(e))

st.write("---")
st.caption("Built with PyTorch & PyTorch Geometric.")

##pip install streamlit torch torch-geometric networkx matplotlib scikit-learn
##python -m streamlit run demo_app.py

