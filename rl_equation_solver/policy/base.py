"""Policy module"""
import logging
import math
import random
from abc import ABC, abstractmethod

import torch
from torch import optim

from rl_equation_solver.config import DefaultConfig

random.seed(42)


logger = logging.getLogger(__name__)


# pylint: disable=not-callable
class BasePolicy(ABC):
    """Base policy class"""

    def __init__(self, env, config=None):
        """
        Parameters
        ----------
        env : Object
            Environment instance.
            e.g. rl_equation_solver.env_linear_equation.Env()
        config : dict | None
            Model configuration. If None then the default model configuration
            in rl_equation_solver.config will be used.
        """
        self.env = env
        self.optimizer = None
        self.policy_network = None
        self.target_network = None
        self.gamma = None
        self.learning_rate = None
        self.tau = None
        self.grad_clip = None
        self.eps_end = None
        self.eps_start = None
        self.eps_decay = None
        self.device = None
        self.eps_decay_steps = None
        self.batch_size = None
        self.epsilon_threshold = None
        self.steps_done = 0
        self._config = config

        self.init_config()

    def init_optimizer(self):
        """Initialize optimizer"""
        self.optimizer = optim.AdamW(
            self.policy_network.parameters(),
            lr=self.learning_rate,
            amsgrad=True,
        )

    def init_config(self) -> None:
        """Initialize model configuration"""
        self.config = DefaultConfig
        if self._config is not None:
            self.config.update(self._config)

        config_log = {}
        for key, val in self.config.items():
            if hasattr(self, key):
                setattr(self, key, val)
                config_log[key] = val
        logger.info(f"Initialized Policy with config: {config_log}")

    def _get_eps_decay(self, steps_done):
        """Get epsilon decay for current number of steps"""
        decay = 0
        if self.eps_decay is None:
            decay = self.eps_start - self.eps_end
            decay *= math.exp(-1.0 * steps_done / self.eps_decay_steps)
        return decay

    def choose_random_action(self):
        """Choose random action rather than the optimal action"""
        return torch.tensor(
            [[self.env.action_space.sample()]],
            device=self.device,
            dtype=torch.long,
        )

    @abstractmethod
    def choose_optimal_action(self, state):
        """
        Choose action with max expected reward :math:`:= max_a Q(s, a)`
        """

    @abstractmethod
    def update_networks(self):
        r"""
        Soft update of network's weights :math:`\theta^{'}
        \leftarrow \tau \theta + (1 - \tau) \theta^{'}`
        """

    @abstractmethod
    def optimize_model(self, loss=None):
        """
        Perform one step of the optimization (on the policy network).
        """

    def get_epsilon_threshold(self, steps_done, training):
        """Get epsilon threshold for eps-greedy routine"""
        epsilon_threshold = self.epsilon_threshold
        if self.epsilon_threshold is None:
            epsilon_threshold = self.eps_end + self._get_eps_decay(steps_done)

            if not training:
                epsilon_threshold = self.eps_end
        return epsilon_threshold

    def choose_action(self, state, training=False):
        """
        Choose action based on given state. Either choose optimal action or
        random action depending on training step.
        """
        random_float = random.random()

        if random_float > self.get_epsilon_threshold(
            self.steps_done, training=training
        ):
            return self.choose_optimal_action(state)
        else:
            return self.choose_random_action()
