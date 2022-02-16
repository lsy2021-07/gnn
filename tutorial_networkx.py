import networkx as nx
import matplotlib.pyplot as plt
G=nx.Graph()
#add one node with level attributes
G.add_node(0,feature=5,label=0)
#get attributes of node 0
node_0_attr = G.nodes[0]
print(node_0_attr)
#add multiple nodes with attributes
G.add_nodes_from([
    (1,{'feature':1,'label':1}),
    (2,{'feature':2,'label':2})
])
#Loop through all the nodes
#set data=True will return node attributes
for node in G.nodes(data=True):
    print(node)

num=G.number_of_nodes()

G.add_edge(0,1,weight=0.5)
edge_0_1_attr=G.edges[(0,1)]
print(edge_0_1_attr)
G.add_edges_from([
    (1,2,{'weight':3}),
    (2,0,{'weight':0.1})
])
for edge in G.edges():
    print(edge)

nx.draw(G,with_labels=True)
plt.show()

node_id=1
print(G.degree[node_id])
for neighbor in G.neighbors(node_id):
    print(neighbor)
