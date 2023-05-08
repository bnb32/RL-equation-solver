"""Off Policy Base Class."""
import math
import random
from typing import Optional, Union

import torch

from rl_equation_solver.agent.base import BaseAgent
from rl_equation_solver.agent.networks import QNetwork
from rl_equation_solver.environment.algebraic import Env
from rl_equation_solver.utilities.utilities import Batch, GraphEmbedding


class OffPolicyAgent(BaseAgent):

    """Off Policy Base Agent."""

    def __init__(self, env: Env, config: Optional[dict] = None) -> None:
        """Initialize the Off Policy Agent.

        Parameters
        ----------
        env : gym.Env
            Environment instance.
            e.g. rl_equation_solver.env_linear_equation.Env()
        config : dict | None
            Model configuration. If None then the default model configuration
            in rl_equation_solver.config will be used.
        """
        self.model: QNetwork
        self.eps_start: float = 0.99
        self.eps_end: float = 0.05
        self.tau: float = 0.005
        self.eps_decay_steps: int = 1000
        self._eps_decay = None
        self._epsilon_threshold = None
        BaseAgent.__init__(self, env, config)

    def update_model(self):
        """Compute loss and update model."""
        if len(self.memory) < self.batch_size:
            return

        loss = self.compute_loss()

        self.env.update_info("loss", loss.item())
        self.env.update_history("loss", loss.item())

        self.step_optimizer(loss)
        self.update_networks()

    def compute_loss(self):
        """Sample memory and compute loss."""
        batch = self.memory.get_batch(self.batch_size, device=self.device)

        values = self.compute_Q(batch)

        true_values = self.compute_expected_Q(batch)

        return self._compute_loss(values, true_values)

    def _get_eps_decay(self, steps_done: int) -> float:
        """Get epsilon decay for current number of steps."""
        decay = 0.0
        if self._eps_decay is None:
            decay = self.eps_start - self.eps_end
            decay *= math.exp(-1.0 * steps_done / self.eps_decay_steps)
        return decay

    def choose_random_action(self) -> torch.Tensor:
        """Choose random action rather than the optimal action."""
        return torch.tensor(
            [[self.env.action_space.sample()]],
            device=self.device,
            dtype=torch.long,
        )

    def get_epsilon_threshold(self, steps_done: int, training: bool) -> float:
        """Get epsilon threshold for eps-greedy routine."""
        epsilon_threshold = self._epsilon_threshold
        if self._epsilon_threshold is None:
            epsilon_threshold = self.eps_end + self._get_eps_decay(steps_done)

            if not training:
                epsilon_threshold = self.eps_end
        return epsilon_threshold

    def choose_action(self, state, training=False):
        """Choose action based on given state. Either choose optimal action or
        random action depending on training step.
        """
        random_float = random.random()

        if random_float > self.get_epsilon_threshold(
            self.steps_done, training=training
        ):
            return self.choose_optimal_action(state)
        else:
            return self.choose_random_action()

    def choose_optimal_action(
        self, state: Union[torch.Tensor, GraphEmbedding]
    ) -> torch.Tensor:
        """Choose action with max expected reward :math:`:= max_a Q(s, a)`.

        max(1) will return largest column value of each row. second column on
        max result is index of where max element was found so we pick action
        with the larger expected reward.
        """
        with torch.no_grad():
            return self.model.policy_network(state).max(1)[1].view(1, 1)

    def compute_expected_Q(self, batch: Batch) -> torch.Tensor:
        r"""Compute the expected Q values.

        Compute according to the Bellman optimality equation
        :math:`Q(s, a) = E(R_{s + 1} + \gamma * max_{a^{'}} Q(s^{'}, a^{'}))`.
        """
        with torch.no_grad():
            out = batch.rewards + self.gamma * (
                1 - batch.dones
            ) * self.compute_next_Q(batch)
        return out

    def compute_next_Q(self, batch: Batch) -> torch.Tensor:
        """Compute next Q values.

        Compute :math:`max_{a} Q(s_{t+1}, a)` for all next states. Expected
        values for next_states are computed based on the "older" target_net;
        selecting their best reward].
        """
        next_vals = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_vals = self.model.target_network(batch.next_states).max(1)[0]

        return next_vals

    def compute_Q(self, batch: Batch) -> torch.Tensor:
        """Compute Q values for "current" states.

        Compute :math:`Q(s_t, a)`. These are the actions which would've been
        taken for each batch state according to policy_net.
        """
        return self.model.policy_network(batch.states).gather(1, batch.actions)

    def update_networks(self) -> None:
        r"""Update the policy and target networks.

        Soft update of the target network's weights :math:`\theta^{'}
        \leftarrow \tau \theta + (1 - \tau) \theta^{'}`
        policy_network.state_dict() returns the parameters of the policy
        network target_network.load_state_dict() loads these parameters into
        the target network.
        """
        target_net_state_dict = self.model.target_network.state_dict()
        policy_net_state_dict = self.model.policy_network.state_dict()
        for key in policy_net_state_dict:
            policy = policy_net_state_dict[key]
            target = target_net_state_dict[key]
            value = target + self.tau * (policy - target)
            target_net_state_dict[key] = value
        self.model.target_network.load_state_dict(target_net_state_dict)
