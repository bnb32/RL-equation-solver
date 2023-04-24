"""A2C implementation"""
import torch
from typing import Union

from rl_equation_solver.agent.networks import LinearActor, LinearCritic
from rl_equation_solver.agent.base import OnPolicyAgent, VectorState, Memory


class Agent(VectorState, OnPolicyAgent):
    """A2C Agent"""

    def __init__(self, policy='MlpPolicy', env=None, config=None):
        OnPolicyAgent.__init__(self, policy, env, config)
        VectorState.__init__(self, env=self.env,
                             n_observations=self.n_observations,
                             n_actions=self.n_actions,
                             device=self.device)
        self.actor = LinearActor(self.n_observations, self.n_actions).to(self.device)
        self.critic = LinearCritic(self.n_observations).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.memory = Memory()
        self.max_steps = 100

    def compute_batch_loss(self) -> Union[torch.Tensor, None]:
        pass
