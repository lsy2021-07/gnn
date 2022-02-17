import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch import nn
from sklearn.decomposition import PCA

# Generate 3 x 4 tensor with all ones

G = nx.karate_club_graph()
print(G.number_of_nodes())
print(G.number_of_edges())
print(list(enumerate(nx.non_edges(G))))
ones = torch.ones(3, 4)
print(ones)

# Generate 3 x 4 tensor with all zeros
zeros = torch.zeros(3, 4)
print(zeros)

# Generate 3 x 4 tensor with random values on the interval [0, 1)
random_tensor = torch.rand(3, 4)
print(random_tensor)

# Get the shape of the tensor
print(ones.shape)

zeros = torch.zeros(3, 4, dtype=torch.float32)
print(zeros.dtype)
zeros = zeros.type(torch.long)
print(zeros.dtype)


# Question 5: Get the edge list of the karate club network and transform it into torch.LongTensor. What is the torch.sum value of pos_edge_index tensor? (10 Points)
def graph_to_edge_list(G):
    # TODO: Implement the function that returns the edge list of
    # an nx.Graph. The returned edge_list should be a list of tuples
    # where each tuple is a tuple representing an edge connected
    # by two nodes.

    edge_list = []

    ############# Your code here ############
    edge_list = [edge for edge in G.edges()]
    print(edge_list)
    #########################################

    return edge_list


def edge_list_to_tensor(edge_list):
    # TODO: Implement the function that transforms the edge_list to
    # tensor. The input edge_list is a list of tuples and the resulting
    # tensor should have the shape [2 x len(edge_list)].

    edge_index = torch.tensor([])

    ############# Your code here ############
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()  # zhuan'zhi
    #########################################

    return edge_index


pos_edge_list = graph_to_edge_list(G)
pos_edge_index = edge_list_to_tensor(pos_edge_list)
print(pos_edge_index)
print("The pos_edge_index tensor has shape {}".format(pos_edge_index.shape))
print("The pos_edge_index tensor has sum value {}".format(torch.sum(pos_edge_index)))

import random


def sample_negative_edges(G, num_neg_samples):
    # TODO: Implement the function that returns a list of negative edges.
    # The number of sampled negative edges is num_neg_samples. You do not
    # need to consider the corner case when the number of possible negative edges
    # is less than num_neg_samples. It should be ok as long as your implementation
    # works on the karate club network. In this implementation, self loops should
    # not be considered as either a positive or negative edge. Also, notice that
    # the karate club network is an undirected graph, if (0, 1) is a positive
    # edge, do you think (1, 0) can be a negative one?

    neg_edge_list = []

    ############# Your code here ############
    neg_edge_list = [random.sample(list(enumerate(nx.non_edges(G))), num_neg_samples)[i][1]
                     for i in range(num_neg_samples)]
    #########################################

    return neg_edge_list


# Sample 78 negative edges
neg_edge_list = sample_negative_edges(G, len(pos_edge_list))

# Transform the negative edge list to tensor
neg_edge_index = edge_list_to_tensor(neg_edge_list)
print("The neg_edge_index tensor has shape {}".format(neg_edge_index))
print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))

# Which of following edges can be negative ones?
edge_1 = (7, 1)
edge_2 = (1, 33)
edge_3 = (33, 22)
edge_4 = (0, 4)
edge_5 = (4, 2)

############# Your code here ############
## Note:
## 1: For each of the 5 edges, print whether it can be negative edge
print(G.has_edge(edge_1[0], edge_1[1]))
print(G.has_edge(edge_2[0], edge_2[1]))
print(G.has_edge(edge_3[0], edge_3[1]))
print(G.has_edge(edge_4[0], edge_4[1]))
print(G.has_edge(edge_5[0], edge_5[1]))
#########################################


# Please do not change / reset the random seed
torch.manual_seed(1)


def create_node_emb(num_node=34, embedding_dim=16):
    # TODO: Implement this function that will create the node embedding matrix.
    # A torch.nn.Embedding layer will be returned. You do not need to change
    # the values of num_node and embedding_dim. The weight matrix of returned
    # layer should be initialized under uniform distribution.

    emb = None

    ############# Your code here ############
    emb = torch.nn.Embedding(num_node, embedding_dim)
    #########################################

    return emb


emb = create_node_emb()
ids = torch.LongTensor([0, 3])

# Print the embedding layer
print("Embedding: {}".format(emb))

# An example that gets the embeddings for node 0 and 3
print(emb(ids))


def visualize_emb(emb):
    X = emb.weight.data.numpy()
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure(figsize=(6, 6))
    club1_x = []
    club1_y = []
    club2_x = []
    club2_y = []
    for node in G.nodes():
        print(node)
    for node in G.nodes(data=True):
        print(node)
        if node[1]['club'] == 'Mr. Hi':
            club1_x.append(components[node[0]][0])
            club1_y.append(components[node[0]][1])
        else:
            club2_x.append(components[node[0]][0])
            club2_y.append(components[node[0]][1])
    plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
    plt.scatter(club2_x, club2_y, color="blue", label="Officer")
    plt.legend()
    plt.show()


# Visualize the initial random embeddding
visualize_emb(emb)

from torch.optim import SGD
import torch.nn as nn


def accuracy(pred, label):
    # TODO: Implement the accuracy function. This function takes the
    # pred tensor (the resulting tensor after sigmoid) and the label
    # tensor (torch.LongTensor). Predicted value greater than 0.5 will
    # be classified as label 1. Else it will be classified as label 0.
    # The returned accuracy should be rounded to 4 decimal places.
    # For example, accuracy 0.82956 will be rounded to 0.8296.

    accu = 0.0

    ############# Your code here ############
    accu=sum(torch.round(pred)==label)/len(pred)
    #########################################

    return accu


def train(emb, loss_fn, sigmoid, train_label, train_edge):
    # TODO: Train the embedding layer here. You can also change epochs and
    # learning rate. In general, you need to implement:
    # (1) Get the embeddings of the nodes in train_edge
    # (2) Dot product the embeddings between each node pair
    # (3) Feed the dot product result into sigmoid
    # (4) Feed the sigmoid output into the loss_fn
    # (5) Print both loss and accuracy of each epoch
    # (6) Update the embeddings using the loss and optimizer
    # (as a sanity check, the loss should decrease during training)

    epochs = 500
    learning_rate = 0.1

    optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)
    print('??')
    for i in range(epochs):
        ############# Your code here ############
        # print('??')
        # optimizer.zero_grad()
        # pred = sigmoid(torch.sum(emb(train_edge[0]).mul(emb(train_edge[1])), 1))
        # print(pred)
        # loss = loss_fn(train_label,pred)
        # loss.backward()
        # optimizer.step()
        # print("Epoch {} Loss: {}, Accuracy: {}".format(i,loss,accuracy(pred)))
        # print("Epoch {} Loss: {}, Accuracy: {}".format(i, loss, accuracy(pred, train_label)))
        optimizer.zero_grad()
        pred = sigmoid(torch.sum(emb(train_edge[0]).mul(emb(train_edge[1])), 1))
        loss = loss_fn(pred, train_label)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        print("Epoch {} Loss: {}, Accuracy: {}".format(i, loss, accuracy(pred, train_label)))
        #########################################
        #########################################


loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()

print(pos_edge_index.shape)

# Generate the positive and negative labels
print(pos_edge_index.shape[1])
pos_label = torch.ones(pos_edge_index.shape[1], )
# temp=torch.ones(4,)
# print(temp)
neg_label = torch.zeros(neg_edge_index.shape[1], )
pos_label = torch.ones(pos_edge_index.shape[1])
# Concat positive and negative labels into one tensor
train_label = torch.cat([pos_label, neg_label], dim=0)

# Concat positive and negative edges into one tensor
# Since the network is very small, we do not split the edges into val/test sets
train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
print(train_edge.shape)

print(train_edge)
print(emb)

print(emb(train_edge[0]))
train(emb, loss_fn, sigmoid, train_label, train_edge)
visualize_emb(emb)