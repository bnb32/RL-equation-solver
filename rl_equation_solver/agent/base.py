"""DQN module."""
import logging
import pickle
from abc import ABC, abstractmethod
from itertools import count
from typing import Optional, Union

import torch
from sympy.core import Expr
from torch import nn, optim

from rl_equation_solver.agent.state import BaseState
from rl_equation_solver.config import DefaultConfig
from rl_equation_solver.environment.algebraic import Env
from rl_equation_solver.utilities.history import ProgressBar
from rl_equation_solver.utilities.loss import LossMixin
from rl_equation_solver.utilities.utilities import GraphEmbedding, Memory

logger = logging.getLogger(__name__)


class BaseAgent(LossMixin, BaseState, ABC):
    """Base Agent without imlemented policy."""

    def __init__(self, env: Env, config: Optional[dict] = None) -> None:
        """Initialize the Agent.

        Parameters
        ----------
        env : gym.Env
            Environment instance.
            e.g. rl_equation_solver.env_linear_equation.Env()
        config : dict | None
            Model configuration. If None then the default model configuration
            in rl_equation_solver.config will be used.
        """
        self.env = env
        self.n_actions: int = env.n_actions
        self.n_observations: int = env.n_obs
        self.optimizer: torch.optim.Optimizer
        self.model: nn.Module
        self._device = None
        self._config = config

        # Configuration properties
        self.batch_size: int = 64
        self.hidden_size: int = 64
        self.memory_cap: int = 10000
        self.state_dim: int = 256
        self.feature_num: int = 100
        self.fill_memory_steps: int = 100
        self.learning_rate: float = 1e-3
        self.gamma: float = 0.99
        self.grad_clip: float = 10
        self.steps_done: int = 0

        self.init_config()

        self.memory = Memory(self.memory_cap)

        logger.info(f"Initialized Agent with device {self.device}")

    @property
    def history(self):
        """Get environment history."""
        return self.env.history

    def init_config(self) -> None:
        """Initialize model configuration."""
        config = DefaultConfig
        if self._config is not None:
            config.update(self._config)

        config_log = {}
        for key, val in config.items():
            if hasattr(self, key):
                setattr(self, key, val)
                config_log[key] = val
        logger.info(f"Initialized Agent with config: {config_log}")

    def update_config(self, config: dict) -> None:
        """Update configuration."""
        self._config = config
        self.init_config()

    @abstractmethod
    def update_model(self) -> None:
        """Compute loss and update the model weights."""

    @abstractmethod
    def choose_action(
        self, state: Union[torch.Tensor, GraphEmbedding], training: bool
    ) -> torch.Tensor:
        """Choose action for given state."""

    @abstractmethod
    def compute_loss(self) -> torch.Tensor:
        """Sample memory and compute loss."""

    @property
    def device(self) -> torch.device:
        """Get device for training network."""
        if self._device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps:0")
            else:
                self._device = torch.device("cpu")
        elif isinstance(self._device, str):
            self._device = torch.device(self._device)
        return self._device

    @device.setter
    def device(self, value: Union[str, torch.device]) -> None:
        """Set device for training network."""
        self._device = value

    def init_optimizer(self) -> None:
        """Initialize optimizer."""
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

    def step(
        self,
        state: Union[torch.Tensor, GraphEmbedding],
        training: bool = False,
    ) -> tuple[Union[torch.Tensor, GraphEmbedding], bool]:
        """Take next step from current state.

        Parameters
        ----------
        state : Expr
            sympy state representation
        episode : int
            Episode number
        training : str
            Whether the step is part of training or inference. Determines
            whether to update the history.

        Returns
        -------
        action : Tensor
            Action taken. Represented as a pytorch tensor.
        next_state : Tensor
            Next state after action. Represented as a pytorch tensor or
            GraphEmbedding.
        done : bool
            Whether solution has been found or if state size conditions have
            been exceeded.
        info : dict
            Dictionary with loss, reward, and state information
        """
        state_string = self.env.state_string
        action = self.choose_action(state, training=training)
        _, reward, done, info = self.env.step(int(action.item()))

        if not done:
            state_string = self.env.state_string
        next_state = self.convert_state(state_string, self.device)

        self.memory.push(
            state,
            next_state,
            action,
            torch.tensor([reward], device=self.device),
            torch.tensor([done], device=self.device),
            info,
        )
        return next_state, done

    def _compute_loss(
        self, values: torch.Tensor, true_values: torch.Tensor
    ) -> torch.Tensor:
        """Compute Huber loss."""
        self.env.loss = self.smooth_l1_loss(values, true_values.unsqueeze(1))
        return self.env.loss

    def step_optimizer(self, loss: Union[torch.Tensor, None]) -> None:
        """Perform one step of the optimization (on the policy network)."""
        if loss is None:
            return

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def fill_memory(self, eval: bool = False) -> None:
        """Fill memory with experiences.

        num_episodes : int
            Number of episodes to fill memory for
        eval : bool
            Whether to run in eval mode - without updating model weights
        """
        if len(self.memory) < self.fill_memory_steps:
            logger.info(f"Filling memory for {self.fill_memory_steps} steps.")
        training = bool(not eval)
        steps = 0
        while len(self.memory) < self.fill_memory_steps:
            state = self.init_state(self.device)
            for _ in count():
                next_state, done = self.step(state, training=training)
                if done:
                    break

                state = next_state
                steps += 1
        if steps > 0:
            self.env.reset_history()

    def learn(
        self, num_episodes: int, eval: bool = False, progress_bar: bool = True
    ) -> None:
        """Fill memory and train model."""
        self.fill_memory(eval=eval)
        self.train(
            num_episodes=num_episodes, eval=eval, progress_bar=progress_bar
        )

    def train(
        self, num_episodes: int, eval: bool = False, progress_bar: bool = True
    ) -> None:
        """Train the model for the given number of episodes.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to train for
        eval : bool
            Whether to run in eval mode - without updating model weights
        progress_bar : bool
            Whether to include a tqdm progress bar.
        """
        logger.info(
            f"Running training routine for {num_episodes} episodes in "
            f"eval={eval} mode."
        )
        self.pbar = ProgressBar(num_episodes, show_progress=progress_bar)
        for _ in range(num_episodes):
            self.run_episode(eval)
            self.pbar.update(1)
        self.pbar.clear()

    def run_episode(self, eval: bool) -> None:
        """Run single episode of training or evaluation. Update the model at
        the update frequency if not running in eval mode.

        Parameters
        ----------
        eval : bool
            Whether to run in eval mode - without updating model weights
        """
        done = False
        training = bool(not eval)
        state = self.init_state(self.device)
        steps = 0
        while not done:
            next_state, done = self.step(state, training=training)
            steps += 1

            if (done or (steps % self.env.update_freq == 0)) and not eval:
                self.update_model()

            self.pbar.update_desc(str(self.env.get_log_info()))

            state = next_state

    def predict(self, state_string: Expr) -> list[Expr]:
        """Predict the solution from the given state_string."""
        state = self.convert_state(state_string, self.device)
        done = False
        steps = 0
        states = [state_string]
        while not done:
            next_state, done = self.step(state, training=False)
            steps += 1
            if next_state is None:
                break
            state = next_state
            states.append(self.env.state_string)
        self.complexity = self.env.get_solution_complexity(
            self.env.state_string
        )
        logger.info(
            f"Final state: {self.env.state_string}. Complexity: "
            f"{self.complexity}. Steps: {steps}."
        )
        return states

    def save(self, model_file: str) -> None:
        """Save the agent."""
        self.close()
        with open(model_file, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved Agent to {model_file}")

    @classmethod
    def load(cls, model_file: str):
        """Load agent from model_file."""
        with open(model_file, "rb") as f:
            agent = pickle.load(f)
        logger.info(f"Loaded agent from {model_file}")
        return agent

    def get_env(self) -> Env:
        """Get environment."""
        return self.env

    def set_env(self, env: Env) -> None:
        """Set the environment."""
        self.env = env

    def close(self) -> None:
        """Close the model."""
        self.pbar.pop()
