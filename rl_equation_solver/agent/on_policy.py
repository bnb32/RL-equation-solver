"""On Policy Base Class."""
from typing import Optional

import numpy as np
import torch

from rl_equation_solver.agent.base import BaseAgent
from rl_equation_solver.agent.networks import ActorCritic
from rl_equation_solver.environment.algebraic import Env


class OnPolicyAgent(BaseAgent):

    """on policy agent class."""

    def __init__(self, env: Env, config: Optional[dict] = None) -> None:
        """Initialize the On Policy Agent.

        Parameters
        ----------
        env : gym.Env
            Environment instance.
            e.g. rl_equation_solver.env_linear_equation.Env()
        config : dict | None
            Model configuration. If None then the default model configuration
            in rl_equation_solver.config will be used.
        """
        self.entropy_coef: float = 0.01
        self.critic_coef: float = 0.5
        self.model: ActorCritic
        self.optimizer: torch.optim.Optimizer
        BaseAgent.__init__(self, env, config)

    def choose_action(self, state, training=False):
        """Choose action based on given state. Either choose optimal action or
        random action depending on training step.
        """
        _, actor_features = self.model.forward(state)
        dist = torch.distributions.Categorical(actor_features)
        return dist.sample()

    def update_model(self) -> None:
        """Compute loss for batch using the stored memory."""
        loss = self.compute_loss()

        self.update_info("loss", loss.item())
        self.update_history("loss", loss.item())

        self.step_optimizer(loss)

        self.memory.clear()

    def compute_loss(self) -> torch.Tensor:
        """Compute total_loss as critic_loss + action_loss - entropy_loss."""
        states, next_states, actions, rewards, dones, _ = self.memory.get_all(
            device=self.device
        )
        values, log_probs, entropy = self.evaluate_actions(states, actions)
        advantages = (
            self.compute_expected_Q(next_states, rewards, dones) - values
        )
        mse = advantages.pow(2).mean()
        critic_loss = self.critic_coef * mse

        actor_loss = (-log_probs * advantages.detach()).mean()
        entropy_loss = (self.entropy_coef * entropy).mean()

        loss = critic_loss + actor_loss - entropy_loss

        return loss

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Evaluate the action for the given state.

        Parameters
        ----------
        states : torch.Tensor
            current state representation
        actions : torch.Tensor
            action for the current state

        Returns
        -------
        value : torch.Tensor
            q value for the current state
        log_probs : torch.Tensor
            log probability for the current state action
        entropy : torch.Tensor
            entropy for current state distribution
        """
        values, actor_features = self.model(states)
        dist = torch.distributions.Categorical(actor_features)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return values, log_probs, entropy

    def compute_expected_Q(
        self,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Bellman update for critic.

        Parameters
        ----------
        next_states : torch.Tensor
            Next state respresentations
        rewards : torch.Tensor
            Reward for current states
        dones : torch.Tensor
            Whether routine is done for the current states

        Returns
        -------
        values : torch.Tensor
            Updated q values
        """
        q_vals = np.zeros((len(self.memory), 1))
        q_val = self.model.get_critic(next_states[-1])
        zip_iter = zip(rewards.flip(dims=[0]), dones.flip(dims=[0]))
        for i, (reward, done) in enumerate(zip_iter):
            q_val = reward + self.gamma * q_val * (1.0 - done.item())
            q_vals[len(self.memory) - 1 - i] = q_val.item()
        return torch.tensor(q_vals, device=self.device)
