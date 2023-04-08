"""Networks for agent policies"""
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class DQN(nn.Module):
    """Simple MLP network."""

    def __init__(self, n_observations, n_actions, hidden_size):
        """
        Parameters
        ----------
        n_observations: int
            observation/state size of the environment
        n_actions : int
            number of discrete actions available in the environment
        hidden_size : int
            size of hidden layers
        """
        super().__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        """
        Forward pass for given state x
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class GCN(nn.Module):
    """Graph Convolution Network"""

    def __init__(self, n_observations, n_actions, hidden_size, dropout=0.1):
        """
        Parameters
        ----------
        n_observations: int
            observation/state size of the environment
        n_actions : int
            number of discrete actions available in the environment
        hidden_size : int
            size of hidden layers
        dropout : float
            dropout rate
        """
        super().__init__()
        self.conv1 = GCNConv(n_observations, hidden_size)
        self.conv2 = GCNConv(hidden_size, n_actions)
        self.dropout = dropout

    def forward(self, data):
        """Forward pass for a given state x"""

        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
