"""A2C implementation"""
import torch

from rl_equation_solver.agent.base import OnPolicyAgent, VectorState
from rl_equation_solver.agent.networks import ActorCritic


class Agent(VectorState, OnPolicyAgent):
    """A2C Agent"""

    def __init__(self, env=None, config=None):
        OnPolicyAgent.__init__(self, env, config)
        VectorState.__init__(
            self,
            env=self.env,
            n_observations=self.n_observations,
            n_actions=self.n_actions,
            device=self.device,
        )
        self.model = ActorCritic(
            self.n_observations, self.n_actions, self.hidden_size
        ).to(self.device)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=self.learning_rate, eps=1e-5
        )
