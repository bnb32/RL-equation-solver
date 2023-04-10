"""DQN module"""
import math
import random
from collections import deque
from itertools import count
import torch
import logging
from abc import abstractmethod
from sympy import simplify, expand
import numpy as np

from rl_equation_solver.config import Config
from rl_equation_solver.utilities.reward import RewardMixin
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
class BaseAgent(RewardMixin, LossMixin):
    """Agent with DQN target and policy networks"""

    def __init__(self, env, hidden_size=Config.HIDDEN_SIZE, device='cpu'):
        """
        Parameters
        ----------
        env : Object
            Environment instance.
            e.g. rl_equation_solver.env_linear_equation.Env()
        hidden_size : int
            size of hidden layers
        """
        self.env = env
        self.hidden_size = hidden_size
        self.steps_done = 0
        self.memory = None
        self.policy_network = None
        self.target_network = None
        self.optimizer = None
        self._history = {}
        self.max_loss = 50
        self.current_episode = 0
        self.info = None
        self._device = device

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

    def choose_optimal_action(self, state):
        """
        Choose action with max expected reward := max_a Q(s, a)

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
        decay = Config.EPSILON_START - Config.EPSILON_END
        decay *= math.exp(-1. * self.steps_done / Config.EPSILON_DECAY)
        epsilon_threshold = Config.EPSILON_END + decay

        self.steps_done += 1
        if random_float > epsilon_threshold or not training:
            return self.choose_optimal_action(state)
        else:
            return self.choose_random_action()

    def compute_loss(self, state_action_values, expected_state_action_values):
        """Compute Huber loss"""
        loss = self.huber_loss(state_action_values,
                               expected_state_action_values.unsqueeze(1))
        return loss

    def update_info(self, key, value):
        """Update history info with given value for the given key"""
        self.info[key] = value

    def choose_random_action(self):
        """Choose random action rather than the optimal action"""
        return torch.tensor([[self.env.action_space.sample()]],
                            device=self.device, dtype=torch.long)

    def optimize_model(self):
        """
        function that performs a single step of the optimization
        """

        if len(self.memory) < Config.BATCH_SIZE:
            return

        transition = self.memory.sample(Config.BATCH_SIZE)
        batch = self.batch_states(transition, device=self.device)

        # Compute Q(s_t, a)
        # These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_network(batch.state_batch) \
            .gather(1, batch.action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed
        # based on the "older" target_net; selecting their best reward with
        # max(1)[0].
        next_state_values = torch.zeros(Config.BATCH_SIZE, device=self.device)

        with torch.no_grad():
            next_state_values[batch.non_final_mask] = \
                self.target_network(batch.non_final_next_states).max(1)[0]

        # Compute the expected Q values
        value = batch.reward_batch + (Config.GAMMA * next_state_values)
        expected_state_action_values = value

        loss = self.compute_loss(state_action_values,
                                 expected_state_action_values)
        self.update_info('loss', loss.item())
        self.update_history('loss', loss.item())

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

    def log_info(self, episode):
        """Write info to logger"""
        out = self.info.copy()
        out['reward'] = '{:.3e}'.format(out['reward'])
        logger.info(f'episode {episode}, {out}')

    def train(self, num_episodes):
        """Train the model for the given number of episodes.

        The agent will perform a soft update of the Target Network's weights,
        with the equation TAU * policy_net_state_dict + (1-TAU) *
        target_net_state_dict, this helps to make the Target Network's weights
        converge to the Policy Network's weights.
        """

        episode_duration = []
        start = self.current_episode
        end = start + num_episodes
        for i in range(start, end):
            state = self.init_state()
            total_reward = 0
            for t in count():
                # sample an action
                action, next_state, done, info = self.step(state,
                                                           episode=i,
                                                           step_number=t,
                                                           training=True)
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

                # Perform one step of the optimization (on the policy network)
                # The agent  performs an optimization step on the Policy
                # Network using the stored memory
                self.optimize_model()

                self.log_info(self.current_episode)
                # Soft update of the target network's weights
                # θ′ ← τθ + (1 − τ)θ′
                # policy_network.state_dict() returns the parameters of the
                # policy network target_network.load_state_dict() loads these
                # parameters into the target network.
                target_net_state_dict = self.target_network.state_dict()
                policy_net_state_dict = self.policy_network.state_dict()
                for key in policy_net_state_dict:
                    value = policy_net_state_dict[key] * Config.TAU
                    value += target_net_state_dict[key] * (1 - Config.TAU)
                    target_net_state_dict[key] = value
                self.target_network.load_state_dict(target_net_state_dict)
                total_reward += info['reward']
                if done:
                    episode_duration.append(t + 1)
                    logger.info(f"Episode {self.current_episode}, Solver "
                                f"terminated after {t} steps with reward "
                                f"{total_reward}. Final state = "
                                f"{self.state_string}")
                    break
            self.current_episode += 1

    @property
    def history(self):
        """Get training history of policy_network"""
        return self._history

    def append_history(self, episode, entry):
        """Append latest step for training history of policy_network"""
        if episode not in self._history:
            self._history[episode] = {'complexity': [], 'loss': [],
                                      'reward': [], 'state': []}
        self._history[episode]['complexity'].append(entry['complexity'])
        self._history[episode]['loss'].append(entry.get('loss', np.nan))
        self._history[episode]['reward'].append(entry['reward'])
        self._history[episode]['state'].append(entry['state'])

    def update_history(self, key, value):
        """Update latest step for training history of policy_network"""
        self._history[self.current_episode][key][-1] = value

    def step(self, state, episode=0, step_number=0, training=False):
        """Take next step from current state

        Parameters
        ----------
        state : str
            State string representation
        episode : int
            Episode number
        step_number : int
            Number of steps taken so far
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
        next_state, done, info = self._step(action.item(), step_number)

        self.append_history(episode, info)

        if done:
            next_state = None

        return action, next_state, done, info

    def expression_complexity(self, state):
        """
        Compute graph / expression complexity for the given state

        Parameters
        ----------
        state : str
            String representation of the current state

        Returns
        -------
        complexity : int
            Number of edges plus number of nodes in graph representation /
            expression_tree of the current solution approximation
        """
        x, *_ = self.env._get_symbols()
        solution_approx = simplify(expand(self.env.equation.replace(x, state)))
        if solution_approx == 0:
            complexity = 0
        else:
            state_graph = utilities.to_graph(solution_approx,
                                             self.env.feature_dict)
            complexity = state_graph.number_of_nodes()
            complexity += state_graph.number_of_edges()

        return complexity

    def _step(self, action: int, step_number: int):
        """
        Take step corresponding to the given action

        Parameters
        ----------
        action : int
            Action index corresponding to the entry in the action list
            constructed in _make_physical_actions
        step_number : int
            Number of steps taken so far.

        Returns
        -------
        new_state : Tensor | GraphEmbedding
            New state after action. Represented as a pytorch Tensor or
            GraphEmbedding
        reward : float
            Reward from taking this step
        done : bool
            Whether problem is solved or if maximum state dimension is reached
        info : dict
            Additional information
        """
        # action is 0,1,2,3, ...,  get the physical actions it indexes
        [operation, term] = self.env.actions[action]
        new_state_string = operation(self.state_string, term)
        new_state_string = simplify(new_state_string)
        new_state_vec = utilities.to_vec(new_state_string,
                                         self.env.feature_dict,
                                         self.env.state_dim)

        # Reward
        reward = self.find_reward(self.state_string, new_state_string)

        # Done
        done = False
        if self.too_long(new_state_vec):
            done = True

        # If complexity is zero, you have solved the problem
        complexity = self.expression_complexity(new_state_string)
        if complexity == 0:
            done = True

        # Update
        self.state_string = new_state_string

        if complexity == 0:
            logger.info(f'solution is: {self.state_string}')

            # reward finding solution in fewer steps
            reward += 10 / (1 + step_number)

        # Extra info
        self.info = {'complexity': complexity, 'loss': np.nan,
                     'reward': reward, 'state': self.state_string}

        return self.convert_state(new_state_string), done, self.info

    def find_reward(self, state_old, state_new):
        """
        Parameters
        ----------
        state_old : str
            String representation of last state
        state_new : str
            String representation of new state

        Returns
        -------
        reward : int
            Difference between loss for state_new and state_old
        """
        return self.diff_loss_reward(state_old, state_new)

    def predict(self, state_string):
        """
        Predict the solution from the given state_string.
        """
        state = self.convert_state(state_string)
        state = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        done = False
        t = 0
        complexities = []
        while not done:
            _, _, _, done = self.step(state, training=False)
            complexity = self.expression_complexity(self.env.state_string)
            complexities.append(complexity)
            t += 1

            if self.is_constant_complexity(complexity):
                done = True

        logger.info(f"Solver terminated after {t} steps. Final "
                    f"state = {self.env.state_string} with complexity = "
                    f"{complexity}.")

    def is_constant_complexity(self, complexities):
        """Check for constant loss over a long number of steps"""
        check = (len(complexities) >= Config.RESET_STEPS
                 and len(set(complexities[-Config.RESET_STEPS:])) <= 1)
        if check:
            logger.info(f'Loss has been constant ({list(complexities)[-1]}) '
                        f'for {Config.RESET_STEPS} steps. Reseting.')
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

    def too_long(self, state):
        """
        Check if state dimension is too large

        Parameters
        ----------
        state : str
            State string representation

        Returns
        -------
        bool
        """
        return len(state) > self.env.state_dim

    @property
    def state_string(self):
        """Get state string representation"""
        return self.env.state_string

    @state_string.setter
    def state_string(self, value):
        """Set state string representation"""
        self.env.state_string = value
