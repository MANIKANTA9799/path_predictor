import streamlit as st 
import networkx as nx
import matplotlib.pyplot as plt
import random
st.title("DEMO FOR LEARNING STREAMLIT")
st.write("lets get started")
num_nodes = st.slider("Select number of nodes", 10, 50, step=10)
prob = st.slider("select probabilty of edge making ",0.5,0.9,step = 0.1)
G = nx.erdos_renyi_graph(num_nodes, prob, seed=42)
for node in G.nodes():
   G.nodes[node]['vul_score']=  round(random.random(), 2)
   G.nodes[node]['critical']= random.choice([0,0,0,1])
colors=[]
for node in G.nodes():
   if (G.nodes[node]['critical']==1):
      colors.append("red")
   else:
      colors.append("green")
pos = nx.spring_layout(G, seed=42)
fig , ax = plt.subplots()
nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax)
st.pyplot(fig)
critical_nodes = []
for node  in G.nodes():
   if (G.nodes[node]['critical']==1):
      critical_nodes.append(node)
   else:
      critical_nodes.append(node)
st.subheader("📍 Critical Nodes")
if (len(critical_nodes)>0):
   st.write(critical_nodes)
else:
   st.write("no critical nodes found")
st.success("done demo")
