"""DQN module"""
from itertools import count
import torch
import logging
from abc import abstractmethod
import numpy as np
import pickle

from rl_equation_solver.config import DefaultConfig
from rl_equation_solver.utilities.loss import LossMixin
from rl_equation_solver.utilities.utilities import ReplayMemory
from rl_equation_solver.policy.base import MlpPolicy


logger = logging.getLogger(__name__)


class BaseAgent(LossMixin, MlpPolicy):
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
        MlpPolicy.__init__(self, env)

        self.env = env
        self.n_actions = env.n_actions
        self.n_observations = env.n_obs
        self.memory = None
        self.optimizer = None
        self._history = {}
        self._device = device
        self._config = config
        self.config = None
        self.solution_steps_max = 10000

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

        self.init_config()

        self.memory = ReplayMemory(self.memory_cap)

    def init_config(self):
        """Initialize model configuration"""
        self.config = DefaultConfig
        if self._config is not None:
            self.config.update(self._config)
        for key, val in self.config.items():
            if hasattr(self, key):
                setattr(self, key, val)
        logger.info(f'Initialized Agent with config: {self.config}')

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

    @property
    def solution_steps(self):
        """Get total number of steps done across all episodes until solution
        is found"""
        return self.env.solution_steps

    @solution_steps.setter
    def solution_steps(self, value):
        """Set total number of steps done across all episodes until solution
        is found"""
        self.env.solution_steps = value

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

    @device.setter
    def device(self, value):
        """Set device for training network"""
        self._device = value

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

        expected_state_action_values = self.compute_expected_Q(batch)

        loss = self.compute_loss(state_action_values,
                                 expected_state_action_values)

        self.update_info('loss', loss.item())
        self.env.update_history('loss', loss.item())

        return loss

    def update_info(self, key, value):
        """Update history info with given value for the given key"""
        self.info[key] = value

    @property
    def run_find_solution(self):
        """Run find_solution while this check is valid"""
        return (np.isnan(self.solution_steps)
                and self.steps_done < self.solution_steps_max)

    def find_solution(self):
        r"""Run the model until the solution is found a single time. This
        can be used for hyperparameter tuning.
        """
        self.history = {}
        self.current_episode = 0
        self.steps_done = 0
        self.solution_steps = np.nan

        while self.run_find_solution:
            self.run_episode(eval=False)

    def fill_memory(self, eval=False):
        """Fill memory with experiences

        num_episodes : int
            Number of episodes to fill memory for
        eval : bool
            Whether to run in eval mode - without updating model weights
        """
        if len(self.memory) < self.fill_memory_steps:
            logger.info(f'Filling memory for {self.fill_memory_steps} steps.')
        training = bool(not eval)
        steps = 0
        while len(self.memory) < self.fill_memory_steps:
            state = self.init_state()
            for _ in count():
                next_state, done = self.add_experience(state,
                                                       training=training)
                if done:
                    break

                state = next_state
                steps += 1
        if steps > 0:
            self.env.reset_history()

    def train(self, num_episodes, eval=False):
        """Train the model for the given number of episodes.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to train for
        eval : bool
            Whether to run in eval mode - without updating model weights
        """
        logger.info(f'Running training routine for {num_episodes} episodes in '
                    f'eval={eval} mode.')
        if eval:
            self.history = {}
            self.current_episode = 0
            self.eps_decay = 0
            self.steps_done = 0

        self.fill_memory()

        start = self.current_episode
        end = start + num_episodes
        for _ in range(start, end):
            self.run_episode(eval)

    def run_episode(self, eval):
        """Run single episode of training or evaluation

        eval : bool
            Whether to run in eval mode - without updating model weights
        """
        training = bool(not eval)
        state = self.init_state()
        for _ in count():
            next_state, done = self.add_experience(state,
                                                   training=training)

            loss = self.compute_batch_loss()

            if not done:
                self.optimize_model(loss)

            if not eval:
                self.update_networks()

            if done:
                self.terminate_msg()
                break

            state = next_state

    def add_experience(self, state, training=True):
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
        reward = torch.tensor([info['reward']], device=self.device)
        self.memory.push(state, action, next_state, reward)
        return next_state, done

    def terminate_msg(self):
        """Log message about solver termination

        Parameters
        ----------
        total_reward : list
            List of reward

        """
        current_episode = list(self.history.keys())[-1]
        total_reward = np.nansum(self.history[current_episode]['reward'])
        mean_loss = np.nanmean(self.history[current_episode]['loss'])
        msg = (f"\nEpisode: {current_episode}, steps_done: {self.steps_done}. "
               f"loop_steps: {self.env.loop_step}, "
               f"total_reward = {total_reward:.3e}, "
               f"mean_loss = {mean_loss:.3e}, "
               f"state = {self.state_string}")
        if not np.isnan(self.env.solution_steps):
            msg += f", solution_steps = {self.env.solution_steps}, "
            logger.info(msg)
        else:
            logger.debug(msg)

    @property
    def history(self):
        """Get training history of the agent"""
        return self.env.history

    @history.setter
    def history(self, value):
        """Set training history of the agent"""
        self.env.history = value

    def predict(self, state_string):
        """
        Predict the solution from the given state_string.
        """
        state = self.convert_state(state_string)
        done = False
        t = 0
        while not done:
            _, _, _, done = self.step(state, training=False)
            complexity = self.env.solution_complexity(self.env.state_string)
            t += 1

        logger.info(f"Solver terminated after {t + 1} steps. Final "
                    f"state = {self.env.state_string} with complexity = "
                    f"{complexity}.")

    # pylint: disable=invalid-unary-operand-type
    def is_constant_complexity(self):
        """Check for constant loss over a long number of steps"""
        current_episode = list(self.history.keys())[-1]
        complexities = self.history[current_episode]['complexity']
        check = (len(complexities) >= self.reset_steps
                 and len(set(complexities[-self.reset_steps:])) <= 1)
        if check:
            logger.info('Complexity has been constant '
                        f'({list(complexities)[-1]}) for {self.reset_steps} '
                        'steps. Reseting.')
        return check

    def save(self, output_file):
        """Save the agent"""
        with open(output_file, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        logger.info(f'Saved Agent to {output_file}')

    @classmethod
    def load(cls, model_file):
        """Load agent from model_file"""
        with open(model_file, 'rb') as f:
            agent = pickle.load(f)
        logger.info(f'Loaded agent from {model_file}')
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

    def get_env(self):
        """Get environment"""
        return self.env

    def set_env(self, env):
        """Set the environment"""
        self.env = env
