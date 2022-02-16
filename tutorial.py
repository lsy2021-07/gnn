import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN,self).__init__()


a=torch.tensor([[1],[3]])
print(a.size())
a=a.unsqueeze(-3)
print(a.size())
