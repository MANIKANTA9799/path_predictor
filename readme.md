# GNN Attack Path Predictor

A beginner-friendly **Graph Neural Network (GNN)** project that predicts potential **attack paths and vulnerable nodes in a computer network**.

This project simulates enterprise network graphs and trains a **Graph Convolutional Network (GCN)** using **PyTorch Geometric** to detect risky nodes that may lie on attacker paths.

An **interactive Streamlit dashboard** is included to visualize predictions and mitigation suggestions.

The repository also includes **learning notebooks used while studying Graph Neural Networks**, so the project shows both **the learning process and the final implementation**.

---

# Project Goals

This project demonstrates how **Graph Neural Networks can model cybersecurity networks** and detect attack paths.

The system:

* Generates synthetic enterprise networks
* Labels nodes that lie on potential attacker paths
* Trains a **GCN model** to predict risky nodes
* Visualizes predictions on a network graph
* Suggests **mitigation strategies**

---

# Technologies Used

* **PyTorch**
* **PyTorch Geometric**
* **NetworkX**
* **Streamlit**
* **Matplotlib**
* **Scikit-learn**
* **NumPy**

---

# Project Structure

```
GNN-Path-Predictor
│
├── data
│   └── sample_network.json
│
├── models
│   └── best_attack_gcn.pt
│
├── notebooks
│   ├── gnn_basics.ipynb
│   ├── graph_basics.ipynb
│   └── data/Planetoid/Cora
│       ├── raw
│       └── processed
│
├── train.py
├── train.ipynb
├── app.py
│
├── requirements.txt
└── README.md
```

---

# Learning Notebooks

The repository includes notebooks used while learning Graph Neural Networks:

### `graph_basics.ipynb`

Introduces fundamental graph concepts using **NetworkX**.

Topics covered:

* Graph creation
* Node and edge features
* Graph visualization
* Shortest paths
* Graph connectivity

---

### `gnn_basics.ipynb`

Introduces **Graph Neural Networks with PyTorch Geometric**.

Topics covered:

* Node embeddings
* Message passing
* Graph Convolutional Networks
* Node classification on the **Cora dataset**

---

### `train.ipynb`

Used for experimenting with training the **Attack Path GCN model** before creating the final `train.py` script.

---

# Dataset Generation

Since real enterprise security graphs are not publicly available, the project **generates synthetic network graphs**.

Each network includes:

* Users
* Workstations
* Servers
* Routers
* Firewalls

Each node has attributes:

* node type
* vulnerability flag
* privilege level
* firewall presence

Edges represent **network communication paths**.

---

# Attack Path Labeling

Nodes are labeled as **risky** if they lie on potential attacker paths.

The labeling process:

1. Select attacker nodes
   (users or workstations)

2. Select target nodes
   (servers)

3. Compute **shortest paths**

4. Mark nodes on those paths as risky

Additional vulnerable nodes may also be labeled randomly.

---

# Model Architecture

The project uses a **Graph Convolutional Network (GCN)**.

Architecture:

```
Node Features
     │
GCN Layer
     │
ReLU
     │
Dropout
     │
GCN Layer
     │
Linear Layer
     │
Risk Score (per node)
```

The model predicts **whether a node lies on a potential attack path**.

Loss function:

```
Binary Cross Entropy with Logits
```

---

# Training

Training generates many synthetic graphs and trains the GCN model.

Example training configuration:

```
Number of graphs : 20
Nodes per graph  : 15
Epochs           : 5
Batch size       : 4
```

Example training output:

```
Epoch 01 | Loss: 0.5761 | Acc: 0.933 | F1: 0.966 | AUC: 0.875
Epoch 02 | Loss: 0.2840 | Acc: 0.933 | F1: 0.966 | AUC: 0.902
Epoch 03 | Loss: 0.1464 | Acc: 0.933 | F1: 0.966 | AUC: 0.960
Epoch 04 | Loss: 0.1864 | Acc: 0.933 | F1: 0.966 | AUC: 0.982
Epoch 05 | Loss: 0.1539 | Acc: 0.933 | F1: 0.966 | AUC: 0.987
```

The best model is saved as:

```
models/best_attack_gcn.pt
```

---

# Streamlit Application

The project includes an interactive **Streamlit dashboard**.

Features:

* Generate synthetic network graphs
* Train a small Temporal GNN demo
* Visualize predicted attack paths
* Identify high-risk nodes
* Recommend mitigation strategies
* Upload custom graph sequences via JSON

Visualization shows:

* Node risk scores
* Ground truth attack paths
* Node importance

---

# Mitigation Recommendation

The system can also suggest **nodes to secure or patch**.

The recommendation strategy considers:

* Model risk score
* Graph centrality
* Attack path disruption

This helps identify **critical nodes to secure in the network**.

---

# Installation

Install dependencies:

```
pip install streamlit torch torch-geometric networkx matplotlib scikit-learn
```

or use:

```
pip install -r requirements.txt
```

---

# Running the Training Script

Train the attack path prediction model:

```
python train.py
```

The trained model will be saved to:

```
models/best_attack_gcn.pt
```

---

# Running the Streamlit App

Launch the interactive dashboard:

```
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

in your browser.

---

# Example Use Case

1. Generate a simulated enterprise network
2. Run the GNN model
3. Identify nodes most likely to be part of attack paths
4. Visualize the network
5. Apply recommended mitigations

---

# Key Concepts Demonstrated

This project demonstrates several important machine learning concepts:

* Graph Neural Networks
* Node classification
* Graph representation learning
* Cybersecurity network modeling
* Synthetic dataset generation
* Model deployment with Streamlit

---

# Future Improvements

Possible extensions include:

* Using real cybersecurity datasets
* Graph Attention Networks (GAT)
* Temporal graph datasets
* Integration with security monitoring tools
* Large-scale enterprise network simulations

---

# Author

This project was built as part of learning **Graph Neural Networks and their applications in cybersecurity** using PyTorch Geometric.

It includes both **learning experiments and a complete working application**.
