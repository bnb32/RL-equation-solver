"""Agent with DQN based policy"""
import torch
import logging

from rl_equation_solver.agent.base import BaseAgent
from rl_equation_solver.agent.networks import DQN
from rl_equation_solver.utilities import utilities


logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """Agent with DQN target and policy networks"""

    def __init__(self, env, config=None, device='cpu'):
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
        super().__init__(env, config, device=device)
        self.policy_network = DQN(self.n_observations, self.n_actions,
                                  self.hidden_size).to(self.device)
        self.target_network = DQN(self.n_observations, self.n_actions,
                                  self.hidden_size).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.init_optimizer()

        logger.info(f'Initialized Agent with device {self.device}')

    def init_state(self):
        """Initialize state as a vector"""
        _ = self.env.reset()
        return torch.tensor(self.env.state_vec, dtype=torch.float32,
                            device=self.device).unsqueeze(0)

    def convert_state(self, state):
        """Convert state string to vector representation"""
        self.env.state_vec = utilities.to_vec(state, self.env.feature_dict,
                                              self.env.state_dim)
        return torch.tensor(self.env.state_vec, dtype=torch.float32,
                            device=self.device).unsqueeze(0)

    def batch_states(self, states, device):
        """Batch agent states"""
        batch = utilities.Batch()(states, device)
        batch.non_final_next_states = torch.cat(batch.non_final_next_states)
        batch.states = torch.cat(batch.states)
        return batch
