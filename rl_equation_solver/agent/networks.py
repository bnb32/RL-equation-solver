"""Networks for agent policies."""
from typing import Union

import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from rl_equation_solver.utilities.utilities import GraphEmbedding


class DQN(nn.Module):
    """Simple MLP network."""

    def __init__(
        self, n_observations: int, n_actions: int, hidden_size: int
    ) -> None:
        """Parameters
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for given state x."""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class GCN(nn.Module):
    """Graph Convolution Network."""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        """Parameters
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
        self.layer1 = GCNConv(
            n_observations, hidden_size, normalize=True, cached=True
        )
        self.layer2 = GCNConv(
            hidden_size, n_actions, normalize=True, cached=True
        )
        self.dropout = dropout

    def _forward(self, graph: nx.Graph) -> torch.Tensor:
        """Forward pass for a given state graph."""
        x = graph.x.T
        edge_index = graph.adj
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.layer1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x, edge_index)
        x = torch.matmul(graph.onehot_values, x)
        return x

    def forward(self, graph: Union[list[nx.Graph], nx.Graph]) -> torch.Tensor:
        """Forward pass for a given state graph or tuple of graphs."""
        if isinstance(graph, (tuple, list)):
            return torch.cat([self._forward(G) for G in graph])
        return self._forward(graph)


class LSTM(nn.Module):
    """LSTM network."""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        hidden_size: int,
        n_features: int,
    ) -> None:
        """Initialize the LSTM network.

        Parameters
        ----------
        n_observations: int
            observation/state size of the environment
        n_actions : int
            number of discrete actions available in the environment
        hidden_size : int
            size of hidden layers
        n_features : int
            Number of features to use in network.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on state x."""
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


class ActorCritic(nn.Module):
    """ActorCritic networks."""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        hidden_size: int,
    ) -> None:
        """Initialize the ActorCritic network.

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
        self.n_obs = n_observations
        self.n_actions = n_actions
        self.hidden_size = hidden_size

        self.feature_size = (
            self.features(torch.zeros(1, 1, self.n_obs)).view(1, -1).size(1)
        )

        self.actor = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, self.n_actions),
            nn.Softmax(dim=1),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def features(self, X) -> torch.Tensor:
        """Get features from input state. This could be a CNN to extract
        features or simply the indentiy operation on the input
        representation.
        """
        return X

    def forward(
        self, X: Union[torch.Tensor, GraphEmbedding]
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass the state through critic and actor networks."""
        x = self.features(X)
        x = x.view(-1, self.feature_size)
        values = self.critic(x)
        actions = self.actor(x)
        return values, actions

    def get_critic(
        self, X: Union[torch.Tensor, GraphEmbedding]
    ) -> torch.Tensor:
        """Get critic output."""
        x = self.features(X)
        x = x.view(-1, self.feature_size)
        return self.critic(x)


class QNetwork(nn.Module):
    """Unified Base QNetwork model with policy and target networks."""

    def __init__(self) -> None:
        """Initialize the network model."""
        super().__init__()
        self.policy_network: nn.Module
        self.target_network: nn.Module

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass the state through policy and target networks."""
        policy = self.policy_network(X)
        target = self.target_network(X)
        return policy, target
