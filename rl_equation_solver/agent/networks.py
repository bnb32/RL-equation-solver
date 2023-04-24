"""Networks for agent policies"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


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
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.layer1 = GCNConv(n_observations, hidden_size, normalize=True, cached=True)
        self.layer2 = GCNConv(hidden_size, n_actions, normalize=True, cached=True)
        self.dropout = dropout

    def _forward(self, graph):
        """Forward pass for a given state graph"""
        x = graph.x.T
        edge_index = graph.adj
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.layer1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x, edge_index)
        x = torch.matmul(graph.onehot_values, x)
        return x

    def forward(self, graph):
        """Forward pass for a given state graph or tuple of graphs"""
        if isinstance(graph, (tuple, list)):
            return torch.cat([self._forward(G) for G in graph])
        else:
            return self._forward(graph)


class LSTM(nn.Module):
    """LSTM network"""

    def __init__(self, n_observations, n_actions, hidden_size, n_features):
        super().__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_observations = n_observations
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=n_observations,
            hidden_size=hidden_size,
            num_layers=n_features,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        """Forward pass on state x"""
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
