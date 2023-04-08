"""Networks for agent policies"""
from torch import nn
import torch.nn.functional as F
from rl_equation_solver.agent.layers import GraphConvolution


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

        self.gc1 = GraphConvolution(n_observations, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, n_actions)
        self.dropout = dropout

    def forward(self, x, adj):
        """Forward pass for a given state x"""

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
