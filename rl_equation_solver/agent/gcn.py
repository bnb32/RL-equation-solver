"""Agent with GCN based policy"""
import logging

from torch import nn

from rl_equation_solver.agent.base import GraphState, OffPolicyAgent
from rl_equation_solver.agent.networks import GCN

logger = logging.getLogger(__name__)


class Model(nn.Module):
    """Unified DQN model with policy and target networks"""

    def __init__(self, n_observations, n_actions, hidden_size, device):
        super().__init__()
        self.policy_network = GCN(n_observations, n_actions, hidden_size).to(device)
        self.target_network = GCN(n_observations, n_actions, hidden_size).to(device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def forward(self, X):
        policy = self.policy_network(X)
        target = self.target_network(X)
        return policy, target


class Agent(GraphState, OffPolicyAgent):
    """Agent with GCN target and policy networks"""

    def __init__(self, env=None, config=None):
        """
        Parameters
        ----------
        env : Object
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
            device=self.device,
        )
        self.model = Model(
            self.n_observations, self.n_actions, self.hidden_size, self.device
        )
        self.init_optimizer()
