"""Agent with DQN based policy"""
import logging

import torch

from rl_equation_solver.agent.base import OffPolicyAgent
from rl_equation_solver.agent.networks import LSTM
from rl_equation_solver.utilities import utilities

logger = logging.getLogger(__name__)


class Agent(OffPolicyAgent):
    """Agent with LSTM target and policy networks"""

    def __init__(self, policy="MlpPolicy", env=None, config=None):
        """
        Parameters
        ----------
        policy : str
            Name of policy to use for the agent. e.g. MlpPolicy
        env : Object
            Environment instance.
            e.g. rl_equation_solver.env_linear_equation.Env()
        config : dict | None
            Model configuration. If None then the default model configuration
            in rl_equation_solver.config will be used.
        device : str
            Device to use for torch objects. e.g. 'cpu' or 'cuda:0'
        """
        super().__init__(policy, env, config)
        self.policy.policy_network = LSTM(
            self.n_observations,
            self.n_actions,
            self.hidden_size,
            self.feature_num,
        ).to(self.device)
        self.policy.target_network = LSTM(
            self.n_observations,
            self.n_actions,
            self.hidden_size,
            self.feature_num,
        ).to(self.device)
        self.policy.target_network.load_state_dict(
            self.policy.policy_network.state_dict()
        )
        self.policy.init_optimizer()
        logger.info(f"Initialized Agent with device {self.device}")

    def init_state(self):
        """Initialize state as a vector"""
        _ = self.env.reset()
        self.env.state_vec = utilities.to_vec(
            self.env.state_string, self.env.feature_dict, self.env.state_dim
        )
        return torch.tensor(
            self.env.state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def convert_state(self, state):
        """Convert state string to vector representation"""
        self.env.state_vec = utilities.to_vec(
            state, self.env.feature_dict, self.env.state_dim
        )
        return torch.tensor(
            self.env.state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def batch_states(self, states, device):
        """Batch agent states"""
        batch = utilities.Batch()(states, device)
        batch.next_states = torch.cat(batch.next_states)
        batch.states = torch.cat(batch.states)
        return batch

    def compute_loss(self, state_action_values, expected_state_action_values):
        """Compute L2 loss"""
        loss = self.l2_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
        return loss
