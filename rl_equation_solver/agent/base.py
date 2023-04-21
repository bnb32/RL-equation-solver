"""DQN module"""
from itertools import count
import torch
import logging
from abc import abstractmethod
import pickle
from tqdm import tqdm
import gym

from rl_equation_solver.config import DefaultConfig
from rl_equation_solver.utilities.loss import LossMixin
from rl_equation_solver.utilities.utilities import ReplayMemory
from rl_equation_solver.policy.base import MlpPolicy
from rl_equation_solver.utilities.history import History


logger = logging.getLogger(__name__)


class BaseAgent(LossMixin, MlpPolicy, History):
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
        History.__init__(self)

        self.env = env
        self.n_actions = env.n_actions
        self.n_observations = env.n_obs
        self.memory = None
        self.optimizer = None
        self._device = device
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
        self._config = config
        self.init_config()

    @abstractmethod
    def init_state(self):
        """Initialize state from the environment. This can be a vector
        representation or graph representation"""

    @abstractmethod
    def convert_state(self, state_string) -> object:
        """Convert state string to appropriate representation. This can be a
        vector representation or graph representation"""

    @abstractmethod
    def batch_states(self, states) -> object:
        """Convert states into a batch"""

    @property
    def device(self) -> torch.device:
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

    def step(self, state,
             training=False) -> tuple[torch.Tensor, object, bool, dict]:
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
        _, _, done, info = self.env.step(action.item())

        if not done:
            state_string = self.state_string
        next_state = self.convert_state(state_string)

        return action, next_state, done, info

    def compute_loss(self, state_action_values,
                     expected_state_action_values) -> torch.Tensor:
        """Compute Huber loss"""
        self.loss = self.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1))
        return self.loss

    def compute_batch_loss(self) -> torch.Tensor:
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
        self.update_history('loss', loss.item())

        return loss

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
            self.reset_history()

    def train(self, num_episodes, eval=False, progress_bar=True):
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

        start = self.current_episode
        end = start + num_episodes

        pbar = None
        if progress_bar:
            pbar = tqdm(num_episodes)
        for _ in range(start, end):
            self.run_episode(eval, pbar=pbar)
            if progress_bar:
                pbar.update(1)
                pbar.set_description(str(self.get_log_info()))

    def run_episode(self, eval: bool, pbar: tqdm = None):
        """Run single episode of training or evaluation

        eval : bool
            Whether to run in eval mode - without updating model weights
        """
        training = bool(not eval)
        state = self.init_state()
        for _ in count():
            next_state, done = self.add_experience(state, training=training)

            loss = self.compute_batch_loss()

            if not done and not eval:
                self.optimize_model(loss)

            if not eval:
                self.update_networks()

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

    def predict(self, state_string):
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
        logger.info(f'Final state: {self.state_string}. Complexity: '
                    f'{self.complexity}. Steps: {steps}.')
        return states

    # pylint: disable=invalid-unary-operand-type
    def is_constant_complexity(self) -> bool:
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

    def get_env(self) -> gym.Env:
        """Get environment"""
        return self.env

    def set_env(self, env):
        """Set the environment"""
        self.env = env
