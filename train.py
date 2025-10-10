# train.py


import os
import random
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


NUM_GRAPHS = 300
NODES_PER_GRAPH = 40
BATCH_SIZE = 16
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED_BASE = 7
os.makedirs("models", exist_ok=True)

def generate_synthetic_network(num_nodes=40, prob_edge=0.06, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    G = nx.erdos_renyi_graph(n=num_nodes, p=prob_edge, seed=seed)
    for i in range(num_nodes):
        if i not in G:
            G.add_node(i)
    type_probs = [('user', 0.45), ('workstation', 0.3), ('server', 0.15), ('router', 0.07), ('firewall', 0.03)]
    types, probs = zip(*type_probs)
    for node in G.nodes():
        t = random.choices(types, probs)[0]
        vuln_prob = {'user':0.05,'workstation':0.12,'server':0.18,'router':0.10,'firewall':0.02}
        is_vuln = 1 if random.random() < vuln_prob[t] else 0
        privilege_map = {'user':0,'workstation':0,'server':2,'router':1,'firewall':2}
        has_firewall = 1 if t == 'firewall' else 0
        G.nodes[node]['type'] = t
        G.nodes[node]['is_vulnerable'] = is_vuln
        G.nodes[node]['privilege'] = privilege_map[t]
        G.nodes[node]['has_firewall'] = has_firewall
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for i in range(len(comps)-1):
            a = random.choice(list(comps[i])); b = random.choice(list(comps[i+1])); G.add_edge(a,b)
    return G

def mark_attack_paths(G, num_attackers=1, num_targets=1, seed=None):
    if seed is not None:
        random.seed(seed)
    nodes = list(G.nodes())
    attacker_pool = [n for n,d in G.nodes(data=True) if d['type'] in ('user','workstation')]
    if not attacker_pool:
        attacker_pool = nodes[:max(1,len(nodes)//4)]
    target_pool = [n for n,d in G.nodes(data=True) if d['type']=='server']
    if not target_pool:
        target_pool = nodes[-max(1,len(nodes)//8):]
    attackers = random.sample(attacker_pool, min(len(attacker_pool), num_attackers))
    targets = random.sample(target_pool, min(len(target_pool), num_targets))
    risky = {n:0 for n in nodes}
    for a in attackers:
        for t in targets:
            try:
                path = nx.shortest_path(G, a, t)
                for n in path:
                    risky[n] = 1
            except nx.NetworkXNoPath:
                pass
    for n,d in G.nodes(data=True):
        if d['is_vulnerable']==1 and random.random() < 0.08:
            risky[n] = 1
    return risky

def nx_to_pyg_data(G, labels_dict):
    TYPE_TO_IDX = {'user':0,'workstation':1,'server':2,'router':3,'firewall':4}
    n = G.number_of_nodes()
    x_list=[]
    for i in range(n):
        d = G.nodes[i]
        onehot=[0]*5
        onehot[TYPE_TO_IDX[d['type']]] = 1
        feat = onehot + [d['is_vulnerable'], d['privilege']/2.0, d['has_firewall']]
        x_list.append(feat)
    x = torch.tensor(x_list, dtype=torch.float)
    edges=[]
    for u,v in G.edges():
        edges.append([u,v]); edges.append([v,u])
    if len(edges)==0:
        edge_index = torch.zeros((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([labels_dict[i] for i in range(n)], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

def create_synthetic_dataset(num_graphs=200, nodes_per_graph=40, seed_base=0):
    dataset=[]
    for i in range(num_graphs):
        G = generate_synthetic_network(num_nodes=nodes_per_graph, prob_edge=0.06, seed=seed_base+i)
        labels = mark_attack_paths(G, num_attackers=1, num_targets=1, seed=seed_base+i)
        data = nx_to_pyg_data(G, labels)
        dataset.append(data)
    return dataset

class AttackGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, 1)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index); x = F.relu(x)
        x = self.lin(x)
        return x.view(-1)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss=0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        p = torch.sigmoid(logits).cpu().numpy()
        pred = (p >= 0.5).astype(int)
        ys.append(batch.y.cpu().numpy()); preds.append(pred); probs.append(p)
    y = np.concatenate(ys); pred = np.concatenate(preds); prob = np.concatenate(probs)
    acc = accuracy_score(y, pred); f1 = f1_score(y, pred, zero_division=0)
    try: auc = roc_auc_score(y, prob)
    except: auc = float('nan')
    return {'accuracy':acc, 'f1':f1, 'auc':auc}


def main():
    print("Device:", DEVICE)
    dataset = create_synthetic_dataset(num_graphs=NUM_GRAPHS, nodes_per_graph=NODES_PER_GRAPH, seed_base=SEED_BASE)
    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_ds = dataset[:split]; test_ds = dataset[split:]
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    in_channels = train_ds[0].num_node_features
    model = AttackGCN(in_channels=in_channels, hidden=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_f1 = -1.0; best_state=None
    for epoch in range(1, EPOCHS+1):
        loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, test_loader, DEVICE)
            print(f"Epoch {epoch:02d} Loss: {loss:.4f} Test Acc: {metrics['accuracy']:.3f} F1: {metrics['f1']:.3f} AUC: {metrics['auc']:.3f}")
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']; best_state = model.state_dict()
        else:
            print(f"Epoch {epoch:02d} Loss: {loss:.4f}")

    if best_state is not None:
        torch.save(best_state, "models/best_attack_gcn.pt"); print("Saved model -> models/best_attack_gcn.pt")

if __name__ == "__main__":
    main()
