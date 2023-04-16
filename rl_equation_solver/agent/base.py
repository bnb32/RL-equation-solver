"""DQN module"""
import math
import random
from collections import deque
from itertools import count
import torch
from torch import optim
import logging
from abc import abstractmethod
from sympy import nsimplify
import numpy as np

from rl_equation_solver.config import DefaultConfig
from rl_equation_solver.utilities.loss import LossMixin
from rl_equation_solver.utilities import utilities


logger = logging.getLogger(__name__)


class ReplayMemory:
    """Stores the Experience Replay buffer"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save the Experience into memory"""
        self.memory.append(utilities.Experience(*args))

    def sample(self, batch_size):
        """select a random batch of Experience for training"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# pylint: disable=not-callable
class BaseAgent(LossMixin):
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
        self.env = env
        self.n_actions = env.n_actions
        self.n_observations = env.n_obs
        self.memory = None
        self.policy_network = None
        self.target_network = None
        self.optimizer = None
        self._history = {}
        self._device = device
        self.config = config

        # Configuration properties
        self.batch_size = None
        self.gamma = None
        self.eps_start = None
        self.eps_end = None
        self.hidden_size = None
        self.eps_decay = None
        self.eps_decay_steps = None
        self.tau = None
        self.learning_rate = None
        self.memory_cap = None
        self.reset_steps = None
        self.vec_dim = None
        self.feature_num = None
        self.grad_clip = None

        self.init_config()

    def init_config(self):
        """Initialize model configuration"""
        config = DefaultConfig
        if self.config is not None:
            config.update(self.config)
        for key, val in config.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def update_config(self, config):
        """Update configuration"""
        self.config = config
        self.init_config()

    @abstractmethod
    def init_state(self):
        """Initialize state from the environment. This can be a vector
        representation or graph representation"""

    @abstractmethod
    def convert_state(self, state_string):
        """Convert state string to appropriate representation. This can be a
        vector representation or graph representation"""

    @abstractmethod
    def batch_states(self, states):
        """Convert states into a batch"""

    def init_optimizer(self):
        """Initialize optimizer"""
        self.optimizer = optim.AdamW(self.policy_network.parameters(),
                                     lr=self.learning_rate, amsgrad=True)

    def _get_eps_decay(self):
        """Get epsilon decay for current number of steps"""
        decay = 0
        if self.eps_decay is None:
            decay = self.eps_start - self.eps_end
            decay *= math.exp(-1. * self.steps_done / self.eps_decay_steps)
        return decay

    @property
    def steps_done(self):
        """Get total number of steps done across all episodes"""
        return self.env.steps_done

    @property
    def current_episode(self):
        """Get current episode"""
        return self.env.current_episode

    @current_episode.setter
    def current_episode(self, value):
        """Set current episode"""
        self.env.current_episode = value

    @property
    def device(self):
        """Get device for training network"""
        if self._device is None:
            if torch.cuda.is_available():
                self._device = torch.device('cuda:0')
            elif torch.backends.mps.is_available():
                self._device = torch.device('mps:0')
            else:
                self._device = torch.device('cpu')
        elif isinstance(self._device, str):
            self._device = torch.device(self._device)
        return self._device

    def step(self, state, training=False):
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
        action = self.choose_action(state, training=training)
        _, _, done, info = self.env.step(action.item())

        if done:
            next_state = None
        else:
            next_state = self.convert_state(self.state_string)

        return action, next_state, done, info

    def choose_optimal_action(self, state):
        """
        Choose action with max expected reward :math:`:= max_a Q(s, a)`

        max(1) will return largest column value of each row. second column on
        max result is index of where max element was found so we pick action
        with the larger expected reward.
        """
        with torch.no_grad():
            return self.policy_network(state).max(1)[1].view(1, 1)

    def choose_action(self, state, training=False):
        """
        Choose action based on given state. Either choose optimal action or
        random action depending on training step.
        """
        random_float = random.random()
        epsilon_threshold = self.eps_end + self._get_eps_decay()

        if not training:
            epsilon_threshold = self.eps_end

        if random_float > epsilon_threshold:
            return self.choose_optimal_action(state)
        else:
            return self.choose_random_action()

    def compute_loss(self, state_action_values, expected_state_action_values):
        """Compute Huber loss"""
        loss = self.smooth_l1_loss(state_action_values,
                                   expected_state_action_values.unsqueeze(1))
        return loss

    def compute_batch_loss(self):
        """Compute loss for batch using the stored memory."""

        if len(self.memory) < self.batch_size:
            return None

        transition = self.memory.sample(self.batch_size)
        batch = self.batch_states(transition, device=self.device)

        state_action_values = self.compute_Q(batch)

        next_state_values = self.compute_V(batch)

        expected_state_action_values = self.compute_expected_Q(
            batch, next_state_values)

        loss = self.compute_loss(state_action_values,
                                 expected_state_action_values)
        return loss

    def update_info(self, key, value):
        """Update history info with given value for the given key"""
        self.info[key] = value

    def choose_random_action(self):
        """Choose random action rather than the optimal action"""
        return torch.tensor([[self.env.action_space.sample()]],
                            device=self.device, dtype=torch.long)

    def optimize_model(self, loss=None):
        """
        Perform one step of the optimization (on the policy network).
        """
        if loss is None:
            return

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(),
                                        self.grad_clip)
        self.optimizer.step()

    def compute_expected_Q(self, batch, next_state_values):
        """
        Compute the expected Q values
        """
        return batch.reward_batch + (self.gamma * next_state_values)

    def compute_V(self, batch):
        """
        Compute :math:`V(s_{t+1})` for all next states. Expected values of
        actions for non_final_next_states are computed based on the "older"
        target_net; selecting their best reward with max(1)[0].
        """
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_state_values[batch.non_final_mask] = \
                self.target_network(batch.non_final_next_states).max(1)[0]

        return next_state_values

    def compute_Q(self, batch):
        """
        Compute :math:`Q(s_t, a)`. These are the actions which would've been
        taken for each batch state according to policy_net
        """
        return self.policy_network(batch.state_batch) \
            .gather(1, batch.action_batch)

    def train(self, num_episodes, eval=False):
        r"""Train the model for the given number of episodes.

        The agent will perform a soft update of the Target Network's weights,
        with the equation :math:`\tau \text{ policy_net_state_dict} +
        (1 - \tau) \text{ target_net_state_dict}`, this helps to make the
        Target Network's weights converge to the Policy Network's weights.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to train for
        eval : bool
            Whether to run in eval mode - without updating model weights
        """
        logger.info(f'Running training routine for {num_episodes} episodes in '
                    f'eval={eval} mode.')
        training = bool(not eval)
        if eval:
            self.history = {}
            self.current_episode = 0

        episode_duration = []
        start = self.current_episode
        end = start + num_episodes
        for i in range(start, end):
            state = self.init_state()
            total_reward = []
            total_loss = []
            for t in count():
                # sample an action
                action, next_state, done, info = self.step(state,
                                                           training=training)
                reward = torch.tensor([info['reward']], device=self.device)

                # Store the experience in the memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Kick agent out of local minima
                check = (self.is_constant_complexity(
                    self.history[i]['complexity']) and not done)
                if check:
                    break

                loss = self.compute_batch_loss()

                if loss is not None:
                    self.update_info('loss', loss.item())
                    self.env.update_history('loss', loss.item())

                self.env.log_info()

                if not done:
                    self.optimize_model(loss)

                if not eval:
                    self.update_networks()

                total_reward.append(info['reward'])
                total_loss.append(info['loss'])
                if done:
                    episode_duration.append(t + 1)
                    msg = (f"Solver terminated after {t} steps with reward "
                           f"{np.nansum(total_reward):.3e} and mean loss "
                           f"{np.nanmean(total_loss):.3e}. Final state = "
                           f"{nsimplify(self.state_string)}")
                    logger.info(msg)
                    break

            self.env.log_info()

    def update_networks(self):
        r"""
        Soft update of the target network's weights :math:`\theta^{'}
        \leftarrow \tau \theta + (1 - \tau) \theta^{'}`
        policy_network.state_dict() returns the parameters of the policy
        network target_network.load_state_dict() loads these parameters into
        the target network.
        """
        target_net_state_dict = self.target_network.state_dict()
        policy_net_state_dict = self.policy_network.state_dict()
        for key in policy_net_state_dict:
            value = policy_net_state_dict[key] * self.tau
            value += target_net_state_dict[key] * (1 - self.tau)
            target_net_state_dict[key] = value
        self.target_network.load_state_dict(target_net_state_dict)

    @property
    def history(self):
        """Get training history of policy_network"""
        return self.env.history

    @history.setter
    def history(self, value):
        """Set training history of policy_network"""
        self.env.history = value

    def predict(self, state_string):
        """
        Predict the solution from the given state_string.
        """
        state = self.convert_state(state_string)
        done = False
        t = 0
        complexities = []
        while not done:
            _, _, _, done = self.step(state, training=False)
            complexity = self.env.expression_complexity(self.env.state_string)
            complexities.append(complexity)
            t += 1

            if self.is_constant_complexity(complexities):
                done = True

        logger.info(f"Solver terminated after {t} steps. Final "
                    f"state = {self.env.state_string} with complexity = "
                    f"{complexity}.")

    # pylint: disable=invalid-unary-operand-type
    def is_constant_complexity(self, complexities):
        """Check for constant loss over a long number of steps"""
        check = (len(complexities) >= self.reset_steps
                 and len(set(complexities[-self.reset_steps:])) <= 1)
        if check:
            logger.info(f'Loss has been constant ({list(complexities)[-1]}) '
                        f'for {self.reset_steps} steps. Reseting.')
        return check

    def save(self, output_file):
        """Save the policy_network"""
        torch.save(self.policy_network.state_dict(), output_file)
        logger.info(f'Saved policy_network to {output_file}')

    @classmethod
    def load(cls, env, model_file):
        """Load policy_network from model_file"""
        agent = cls(env)
        agent.policy_network.load_state_dict(torch.load(model_file))
        logger.info(f'Loaded policy_network from {model_file}')
        return agent

    @property
    def state_string(self):
        """Get state string representation"""
        return self.env.state_string

    @state_string.setter
    def state_string(self, value):
        """Set state string representation"""
        self.env.state_string = value

    @property
    def info(self):
        """Get environment info"""
        return self.env.info

    @info.setter
    def info(self, value):
        """Set environment info"""
        self.env.info = value
