"""State Representations."""
from abc import ABC, abstractmethod
from typing import Union

import torch
from sympy.core import Expr

from rl_equation_solver.environment.algebraic import Env
from rl_equation_solver.utilities import utilities
from rl_equation_solver.utilities.utilities import Batch, GraphEmbedding


class BaseState(ABC):
    """Base state class."""

    @abstractmethod
    def init_state(
        self, device: torch.device
    ) -> Union[torch.Tensor, GraphEmbedding]:
        """Initialize state from the environment. This can be a vector
        representation or graph representation.
        """

    @abstractmethod
    def convert_state(
        self, state: Expr, device: torch.device
    ) -> Union[torch.Tensor, GraphEmbedding]:
        """Convert state string to appropriate representation.

        This can be a vector representation or graph representation.
        """

    @abstractmethod
    def batch_states(
        self,
        states: list[Union[torch.Tensor, GraphEmbedding]],
        device: torch.device,
    ) -> Batch:
        """Convert states into a batch."""


class VectorState(BaseState):
    """Class for vector state representation."""

    def __init__(self, env: Env, n_observations: int, n_actions: int) -> None:
        """Initialize the vector state representation."""
        self.env = env
        self.n_observations = n_observations
        self.n_actions = n_actions

    def init_state(self, device):
        """Initialize state as a vector."""
        _ = self.env.reset()
        return torch.tensor(
            self.env.state_vec, dtype=torch.float32, device=device
        ).unsqueeze(0)

    def convert_state(self, state, device):
        """Convert state string to vector representation."""
        self.env.state_vec = utilities.to_vec(
            state, self.env.feature_dict, self.env.state_dim
        )
        return torch.tensor(
            self.env.state_vec, dtype=torch.float32, device=device
        ).unsqueeze(0)

    def batch_states(self, states, device):
        """Batch agent states."""
        batch = Batch()(states, device)
        batch.next_states = torch.cat(batch.next_states)
        batch.states = torch.cat(batch.states)
        return batch


class GraphState(BaseState):
    """Class for graph state representation."""

    def __init__(
        self,
        env: Env,
        n_observations: int,
        n_actions: int,
        feature_num: int,
    ) -> None:
        """Initialize the graph state representation."""
        self.env = env
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.feature_num = feature_num

    def init_state(self, device):
        """Initialize state as a graph."""
        _ = self.env.reset()
        self.env.graph = utilities.to_graph(
            self.env.state_string, self.env.feature_dict
        )
        return GraphEmbedding(
            self.env.graph,
            n_observations=self.n_observations,
            n_features=self.feature_num,
            device=device,
        )

    def convert_state(self, state, device):
        """Convert state string to graph representation."""
        self.env.graph = utilities.to_graph(state, self.env.feature_dict)
        return GraphEmbedding(
            self.env.graph,
            n_observations=self.n_observations,
            n_features=self.feature_num,
            device=device,
        )

    def batch_states(self, states, device):
        """Batch agent states."""
        return Batch()(states, device)
