"""DQN module"""
import logging
import math
import pickle
import random
from abc import ABC, abstractmethod
from itertools import count
from typing import Union

import gym
import torch
from torch import optim
from tqdm import tqdm

from rl_equation_solver.config import DefaultConfig
from rl_equation_solver.utilities import utilities
from rl_equation_solver.utilities.history import History
from rl_equation_solver.utilities.loss import LossMixin
from rl_equation_solver.utilities.utilities import GraphEmbedding, Memory

logger = logging.getLogger(__name__)


class BaseAgent(LossMixin, History, ABC):
    """Base Agent without imlemented policy"""

    def __init__(self, env=None, config=None):
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
        History.__init__(self)

        self.n_actions = env.n_actions
        self.n_observations = env.n_obs
        self.memory = None
        self.optimizer = None
        self._device = None
        self._config = config
        self.config = None
        self.model = None

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
        self.grad_clip = None
        self.tau = None
        self.grad_clip = None
        self.eps_end = None
        self.eps_start = None
        self.eps_decay = None
        self.eps_decay_steps = None
        self.epsilon_threshold = None
        self.steps_done = 0
        self.entropy_coef = None
        self.critic_coef = None

        self.init_config()

        logger.info(f"Initialized Agent with device {self.device}")

        self.memory = Memory(self.memory_cap)

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
    def init_state(self) -> Union[torch.Tensor, GraphEmbedding]:
        """Initialize state from the environment. This can be a vector
        representation or graph representation"""

    @abstractmethod
    def convert_state(self, state_string) -> Union[torch.Tensor, GraphEmbedding]:
        """Convert state string to appropriate representation. This can be a
        vector representation or graph representation"""

    @abstractmethod
    def batch_states(self, states, device) -> utilities.Batch:
        """Convert states into a batch"""

    @abstractmethod
    def update_model(self, done: bool, eval: bool) -> None:
        """Compute loss and update the model weights"""

    @abstractmethod
    def choose_action(
        self, state: Union[torch.Tensor, GraphEmbedding], training: bool
    ) -> torch.Tensor:
        """Choose action for given state"""

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

    def init_optimizer(self):
        """Initialize optimizer"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            amsgrad=True,
        )

    def optimize_model(self, loss: Union[torch.Tensor, None]):
        """
        Perform one step of the optimization (on the policy network).
        """
        if loss is None:
            return

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def step(
        self, state, training: bool = False
    ) -> tuple[Union[torch.Tensor, GraphEmbedding], bool]:
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
        action = self.choose_action(state, training=training)
        _, reward, done, info = self.env.step(action.item())

        if not done:
            state_string = self.state_string
        next_state = self.convert_state(state_string)

        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)
        self.memory.push(state, next_state, action, reward, done, info)
        return next_state, done

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
                next_state, done = self.step(state, training=training)
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
            next_state, done = self.step(state, training=training)

            self.update_model(done=done, eval=eval)

            if done:
                break

            state = next_state

            if pbar is not None:
                pbar.set_description(str(self.get_log_info()))

    def predict(self, state_string) -> list[object]:
        """
        Predict the solution from the given state_string.
        """
        state = self.convert_state(state_string)
        done = False
        steps = 0
        states = [state_string]
        while not done:
            next_state, done = self.step(state, training=False)
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

    def update_model(self, done: bool, eval: bool) -> None:
        """Compute loss and update model"""

        if len(self.memory) < self.batch_size:
            return

        transition = self.memory.sample(self.batch_size)
        batch = self.batch_states(transition, device=self.device)

        state_action_values = self.compute_Q(batch)

        expected_state_action_values = self.compute_expected_Q(batch)

        loss = self.compute_loss(state_action_values, expected_state_action_values)

        self.update_info("loss", loss.item())
        self.update_history("loss", loss.item())

        if not done and not eval:
            self.optimize_model(loss)

        if not eval:
            self.update_networks()

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

    def choose_optimal_action(self, state):
        """
        Choose action with max expected reward :math:`:= max_a Q(s, a)`

        max(1) will return largest column value of each row. second column on
        max result is index of where max element was found so we pick action
        with the larger expected reward.
        """
        with torch.no_grad():
            return self.model.policy_network(state).max(1)[1].view(1, 1)

    def compute_expected_Q(self, batch):
        r"""
        Compute the expected Q values according to the Bellman optimality
        equation :math:`Q(s, a) = E(R_{s + 1} + \gamma *
        max_{a^{'}} Q(s^{'}, a^{'}))`
        """
        with torch.no_grad():
            out = batch.rewards
            out += self.gamma * torch.mul(1 - batch.dones, self.compute_next_Q(batch))
        return out

    def compute_next_Q(self, batch):
        """
        Compute :math:`max_{a} Q(s_{t+1}, a)` for all next states. Expected
        values for next_states are computed based on the "older" target_net;
        selecting their best reward].
        """
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_state_values = self.model.target_network(batch.next_states).max(1)[0]

        return next_state_values

    def compute_Q(self, batch):
        """
        Compute :math:`Q(s_t, a)`. These are the actions which would've
        been taken for each batch state according to policy_net
        """
        return self.model.policy_network(batch.states).gather(1, batch.actions)

    def update_networks(self):
        r"""
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


class OnPolicyAgent(BaseAgent):
    """on policy agent class"""

    def choose_action(self, state, training=False):
        """
        Choose action based on given state. Either choose optimal action or
        random action depending on training step.
        """
        probs = self.model.actor(state)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return action

    def update_model(self, done: bool, eval: bool) -> None:
        """Compute loss for batch using the stored memory."""
        states, next_states, actions, rewards, dones, _ = self.memory.pop_all(
            device=self.device
        )
        true_values = self.compute_expected_Q(next_states, rewards, dones)
        values, log_probs, entropy = self.evaluate_action(states, actions)
        advantages = true_values - values
        critic_loss = advantages.pow(2).mean()
        actor_loss = -(log_probs * advantages)
        total_loss = self.critic_coef * critic_loss + actor_loss
        total_loss -= self.entropy_coef * entropy

        self.update_info("loss", total_loss.item())
        self.update_history("loss", total_loss.item())

        if not done:
            self.optimize_model(total_loss)

    def evaluate_action(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the action for the given state

        Parameters
        ----------
        state : torch.Tensor
            current state representation
        action : torch.Tensor
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
        value, probs = self.model.forward(state)
        dist = torch.distributions.Categorical(probs=probs)

        log_probs = dist.log_prob(action).view(-1, 1)
        entropy = dist.entropy().mean()

        return value, log_probs, entropy

    def compute_expected_Q(
        self, next_state: torch.Tensor, reward: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        """Bellman update for critic

        Parameters
        ----------
        next_state : torch.Tensor
            Next state respresentation
        reward : torch.Tensor
            Reward for current state
        done : torch.Tensor
            Whether routine is done for the current state

        Returns
        -------
        value : torch.Tensor
            Updated q value
        """
        next_value = self.model.critic(next_state)
        return reward + (1 - int(done.item())) * self.gamma * next_value
