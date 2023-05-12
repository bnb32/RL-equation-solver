"""Agent with GCN based policy."""
import logging
from typing import Optional

import torch

from rl_equation_solver.agent.networks import GCN, QNetwork
from rl_equation_solver.agent.off_policy import OffPolicyAgent
from rl_equation_solver.agent.state import GraphState
from rl_equation_solver.environment.algebraic import Env

logger = logging.getLogger(__name__)


class Model(QNetwork):
    """Unified DQN model with policy and target networks."""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        hidden_size: int,
        device: torch.device,
    ) -> None:
        """Initialize the network model."""
        super().__init__()
        self.policy_network = GCN(n_observations, n_actions, hidden_size).to(
            device
        )
        self.target_network = GCN(n_observations, n_actions, hidden_size).to(
            device
        )
        self.target_network.load_state_dict(self.policy_network.state_dict())


class Agent(GraphState, OffPolicyAgent):
    """Agent with GCN target and policy networks."""

    def __init__(self, env: Env, config: Optional[dict] = None) -> None:
        """Parameters
        ----------
        env : gym.Env
            Environment instance.
            e.g. rl_equation_solver.env_linear_equation.Env()
        config : dict | None
            Model configuration. If None then the default model configuration
            in rl_equation_solver.config will be used.
        device : str
            Device to use for torch objects. e.g. 'cpu' or 'cuda:0'
        """
        OffPolicyAgent.__init__(self, env, config)
        GraphState.__init__(
            self,
            env=self.env,
            n_observations=self.n_observations,
            n_actions=self.n_actions,
            feature_num=self.feature_num,
        )
        self.model = Model(
            self.n_observations, self.n_actions, self.hidden_size, self.device
        )
        self.init_optimizer()
