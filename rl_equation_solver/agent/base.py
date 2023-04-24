"""DQN module"""
import logging
import pickle
from abc import ABC, abstractmethod
from itertools import count
from typing import Union
import numpy as np

import gym
import torch
from tqdm import tqdm

import rl_equation_solver.policy.policies
from rl_equation_solver.config import DefaultConfig
from rl_equation_solver.utilities import utilities
from rl_equation_solver.utilities.history import History, Info
from rl_equation_solver.utilities.loss import LossMixin
from rl_equation_solver.utilities.utilities import ReplayMemory
from rl_equation_solver.utilities.utilities import GraphEmbedding

logger = logging.getLogger(__name__)


class BaseAgent(LossMixin, History, ABC):
    """Base Agent without imlemented policy"""

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
        """
        self.env = env

        self.policy = getattr(rl_equation_solver.policy.policies, policy)
        self.policy = self.policy(env)
        History.__init__(self)

        self.n_actions = env.n_actions
        self.n_observations = env.n_obs
        self.memory = None
        self.optimizer = None
        self._device = None
        self._config = config
        self.config = None

        # Configuration properties
        self.batch_size = None
        self.eps_start = None
        self.eps_end = None
        self.hidden_size = None
        self.memory_cap = None
        self.reset_steps = None
        self.vec_dim = None
        self.feature_num = None
        self.fill_memory_steps = None
        self.learning_rate = None
        self.gamma = None

        self.init_config()

        self.memory = ReplayMemory(self.memory_cap)

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
        logger.info(f"Initialized Agent with config: {config_log}")

    def update_config(self, config) -> None:
        """Update configuration"""
        self._config = config
        self.init_config()

    @abstractmethod
    def init_state(self) -> object:
        """Initialize state from the environment. This can be a vector
        representation or graph representation"""

    @abstractmethod
    def convert_state(self, state_string) -> object:
        """Convert state string to appropriate representation. This can be a
        vector representation or graph representation"""

    @abstractmethod
    def batch_states(self, states, device) -> utilities.Batch:
        """Convert states into a batch"""

    @abstractmethod
    def compute_batch_loss(self) -> Union[torch.Tensor, None]:
        """Compute loss for batch using the stored memory."""

    @property
    def device(self) -> torch.device:
        """Get device for training network"""
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
    def device(self, value):
        """Set device for training network"""
        self._device = value

    def step(self, state, training=False) -> tuple[torch.Tensor, object, bool, Info]:
        """Take next step from current state

        Parameters
        ----------
        state : str
            State string representation
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
        state_string = self.state_string
        action = self.policy.choose_action(state, training=training)
        _, _, done, info = self.env.step(action.item())

        if not done:
            state_string = self.state_string
        next_state = self.convert_state(state_string)

        return action, next_state, done, info

    def compute_loss(
        self, state_action_values, expected_state_action_values
    ) -> torch.Tensor:
        """Compute Huber loss"""
        self.loss = self.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
        return self.loss

    def fill_memory(self, eval=False) -> None:
        """Fill memory with experiences

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
            state = self.init_state()
            for _ in count():
                next_state, done = self.add_experience(state, training=training)
                if done:
                    break

                state = next_state
                steps += 1
        if steps > 0:
            self.reset_history()

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
        """
        logger.info(
            f"Running training routine for {num_episodes} episodes in "
            f"eval={eval} mode."
        )

        start = self.current_episode
        end = start + num_episodes

        pbar = tqdm(num_episodes) if progress_bar else None
        for _ in range(start, end):
            self.run_episode(eval, pbar=pbar)
            if progress_bar:
                pbar.update(1)
                pbar.set_description(str(self.get_log_info()))

    def run_episode(self, eval: bool, pbar: tqdm = None) -> None:
        """Run single episode of training or evaluation

        eval : bool
            Whether to run in eval mode - without updating model weights
        """
        done = False
        training = bool(not eval)
        state = self.init_state()
        while not done:
            next_state, done = self.add_experience(state, training=training)

            loss = self.compute_batch_loss()

            if not done and not eval:
                self.policy.optimize_model(loss)

            if not eval:
                self.policy.update_networks()

            if done:
                break

            state = next_state

            if pbar is not None:
                pbar.set_description(str(self.get_log_info()))

    def add_experience(self, state, training=True) -> tuple[object, bool]:
        """Add experience to the memory for a given starting state

        Parameters
        ----------
        state : sympy.expr
            Symbolic expression of current approximate solution
        training : bool
            Whether this during training or evaluation

        Returns
        -------
        next_state : sympy.expr
            Symbolic expression for next approximate solution
        done : bool
            Whether the solution is exact / state vector size exceeded limit or
            if training / evaluation should continue
        """
        action, next_state, done, info = self.step(state, training=training)
        reward = torch.tensor([info.reward], device=self.device)
        self.memory.push(state, next_state, action, reward, done, info)
        return next_state, done

    def predict(self, state_string) -> list[object]:
        """
        Predict the solution from the given state_string.
        """
        state = self.convert_state(state_string)
        done = False
        steps = 0
        states = [state_string]
        while not done:
            _, next_state, done, _ = self.step(state, training=False)
            steps += 1
            if next_state is None:
                break
            state = next_state
            states.append(self.state_string)
        self.complexity = self.env.get_solution_complexity(self.state_string)
        logger.info(
            f"Final state: {self.state_string}. Complexity: "
            f"{self.complexity}. Steps: {steps}."
        )
        return states

    # pylint: disable=invalid-unary-operand-type
    def is_constant_complexity(self) -> bool:
        """Check for constant loss over a long number of steps"""
        complexities = self.history[-1]["complexity"]
        check = (
            len(complexities) >= self.reset_steps
            and len(set(complexities[-self.reset_steps :])) <= 1
        )
        if check:
            logger.info(
                "Complexity has been constant "
                f"({list(complexities)[-1]}) for {self.reset_steps} "
                "steps. Reseting."
            )
        return check

    def save(self, output_file):
        """Save the agent"""
        with open(output_file, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved Agent to {output_file}")

    @classmethod
    def load(cls, model_file) -> object:
        """Load agent from model_file"""
        with open(model_file, "rb") as f:
            agent = pickle.load(f)
        logger.info(f"Loaded agent from {model_file}")
        return agent

    @property
    def state_string(self) -> object:
        """Get state string representation"""
        return self.env.state_string

    @state_string.setter
    def state_string(self, value):
        """Set state string representation"""
        self.env.state_string = value

    def get_env(self) -> gym.Env:
        """Get environment"""
        return self.env

    def set_env(self, env):
        """Set the environment"""
        self.env = env


class OffPolicyAgent(BaseAgent):
    """Off Policy Base Agent"""

    def compute_batch_loss(self) -> Union[torch.Tensor, None]:
        """Compute loss for batch using the stored memory."""

        if len(self.memory) < self.batch_size:
            return None

        transition = self.memory.sample(self.batch_size)
        batch = self.batch_states(transition, device=self.device)

        state_action_values = self.policy.compute_Q(batch)

        expected_state_action_values = self.policy.compute_expected_Q(batch)

        loss = self.compute_loss(state_action_values, expected_state_action_values)

        self.update_info("loss", loss.item())
        self.update_history("loss", loss.item())

        return loss


class VectorState:
    """Class for vector state representation"""

    def __init__(self, env, n_observations, n_actions, device):
        self.env = env
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.device = device

    def init_state(self) -> torch.Tensor:
        """Initialize state as a vector"""
        _ = self.env.reset()
        return torch.tensor(
            self.env.state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def convert_state(self, state) -> torch.Tensor:
        """Convert state string to vector representation"""
        self.env.state_vec = utilities.to_vec(
            state, self.env.feature_dict, self.env.state_dim
        )
        return torch.tensor(
            self.env.state_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def batch_states(self, states, device) -> utilities.Batch:
        """Batch agent states"""
        batch = utilities.Batch()(states, device)
        batch.next_states = torch.cat(batch.next_states)
        batch.states = torch.cat(batch.states)
        return batch


class GraphState:
    """Class for graph state representation"""

    def __init__(self, env, n_observations, n_actions, feature_num, device):
        self.env = env
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.feature_num = feature_num
        self.device = device

    def init_state(self) -> GraphEmbedding:
        """Initialize state as a graph"""
        _ = self.env.reset()
        self.env.graph = utilities.to_graph(
            self.env.state_string, self.env.feature_dict
        )
        return GraphEmbedding(
            self.env.graph,
            n_observations=self.n_observations,
            n_features=self.feature_num,
            device=self.device,
        )

    def convert_state(self, state) -> GraphEmbedding:
        """Convert state string to graph representation"""
        self.env.graph = utilities.to_graph(state, self.env.feature_dict)
        return GraphEmbedding(
            self.env.graph,
            n_observations=self.n_observations,
            n_features=self.feature_num,
            device=self.device,
        )

    def batch_states(self, states, device) -> utilities.Batch:
        """Batch agent states"""
        batch = utilities.Batch()(states, device)
        return batch


class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def push(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.rewards)


class OnPolicyAgent(BaseAgent):

    # train function
    def optimize_model(self, q_val):
        values = torch.stack(self.memory.values)
        q_vals = np.zeros((len(self.memory), 1))

        # target values are calculated backward
        # it's super important to handle correctly done states,
        # for those cases we want our to target to be equal to the reward only
        for i, (_, _, reward, done) in enumerate(self.memory.reversed()):
            q_val = reward + self.gamma * q_val * (1.0 - done)
            # store values from the end to the beginning
            q_vals[len(self.memory) - 1 - i] = q_val.item()

        advantage = torch.tensor(q_vals, device=self.device) - values

        critic_loss = advantage.pow(2).mean()
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        actor_loss = (-torch.stack(self.memory.log_probs)*advantage.detach()).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        self.update_info('loss', critic_loss.item())
        self.update_history("loss", critic_loss.item())

    def run_episode(self, eval=False, pbar=None):
        done = False
        state = self.init_state()

        training = bool(not eval)
        steps = 0
        while not done:
            _, next_state, done, _ = self.step(state, training=training)
            steps += 1
            state = next_state

            # train if done or num steps > max_steps
            if done or (steps % self.max_steps == 0):
                last_q_val = self.critic(next_state)
                self.optimize_model(last_q_val)
                self.memory.clear()

            if pbar is not None:
                pbar.set_description(str(self.get_log_info()))

    def step(self, state, training=False) -> tuple[torch.Tensor, object, bool, Info]:
        """Take next step from current state

        Parameters
        ----------
        state : str
            State string representation
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
        state_string = self.state_string
        dist, action = self.choose_action(state)
        next_state, reward, done, info = self.env.step(action.item())
        self.memory.push(dist.log_prob(action), self.critic(state), reward, done)

        if not done:
            state_string = self.state_string
        next_state = self.convert_state(state_string)

        return action, next_state, done, info

    def choose_action(self, state):
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return dist, action
