"""DQN module"""
import math
import random
from collections import namedtuple, deque
from itertools import count
import torch
from torch import nn
import logging
from abc import abstractmethod

from rl_equation_solver.config import Config


logger = logging.getLogger(__name__)


# structure of the Experiences to store
Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """Stores the Experience Replay buffer"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save the Experience into memory"""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """select a random batch of Experience for training"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# pylint: disable=not-callable
class BaseAgent:
    """Agent with DQN target and policy networks"""

    def __init__(self, env, hidden_size=Config.HIDDEN_SIZE):
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

    @abstractmethod
    def init_state(self):
        """Initialize state from the environment. This can be a vector
        representation or graph representation"""

    @abstractmethod
    def convert_state(self, state_string):
        """Convert state string to appropriate representation. This can be a
        vector representation or graph representation"""

    @property
    def device(self):
        """Get device for training network"""
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            return torch.device('mps:0')
        else:
            return torch.device('cpu')

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
        decay = (Config.EPSILON_START - Config.EPSILON_END)
        decay *= math.exp(-1. * self.steps_done / Config.EPSILON_DECAY)
        epsilon_threshold = Config.EPSILON_END + decay

        self.steps_done += 1
        if random_float > epsilon_threshold or not training:
            return self.choose_optimal_action(state)
        else:
            return self.choose_random_action()

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
        batch = Experience(*zip(*transition))

        # Compute a mask of non-final states and concatenate the batch element
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        # These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = \
            self.policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed
        # based on the "older" target_net; selecting their best reward with
        # max(1)[0].
        next_state_values = torch.zeros(Config.BATCH_SIZE, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = \
                self.target_network(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        value = reward_batch + (Config.GAMMA * next_state_values)
        expected_state_action_values = value

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

    def train(self, num_episodes):
        """Train the model for the given number of episodes.

        The agent will perform a soft update of the Target Network's weights,
        with the equation TAU * policy_net_state_dict + (1-TAU) *
        target_net_state_dict, this helps to make the Target Network's weights
        converge to the Policy Network's weights.
        """

        episode_duration = []
        for i in range(num_episodes):
            # At the beginning we reset the environment an initialize the
            # state Tensor.
            state = self.init_state()
            state = torch.tensor(state, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            total_reward = 0
            for t in count():
                # sample an action
                action, next_state, done, info = self.step(state,
                                                           episode=i,
                                                           step_number=t,
                                                           training=True)

                logger.info(f'episode {i}, info = {info}')

                reward = torch.tensor([info['reward']], device=self.device)

                # Store the experience in the memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Kick agent out of local minima
                if self.is_constant_loss(self.history[i]['loss']):
                    break

                # Perform one step of the optimization (on the policy network)
                # The agent  performs an optimization step on the Policy
                # Network using the stored memory
                self.optimize_model()

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
                    logger.info(f"Episode {i}, Solver terminated after {t} "
                                f"steps with reward {total_reward}. Final "
                                f"state = {self.env.state_string}")
                    break

    @property
    def history(self):
        """Get training history of policy_network"""
        return self._history

    def update_history(self, episode, entry):
        """Update training history of policy_network"""
        if episode not in self._history:
            self._history[episode] = {'loss': [], 'reward': [], 'state': []}
        self._history[episode]['loss'].append(entry['loss'])
        self._history[episode]['reward'].append(entry['reward'])
        self._history[episode]['state'].append(entry['state'])

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
        next_state : Tensor | None
            Next state after action. Represented as a pytorch tensor.
        done : bool
            Whether solution has been found or if state size conditions have
            been exceeded.
        info : dict
            Dictionary with loss, reward, and state information
        """
        action = self.choose_action(state, training=training)
        observation, done, info = self.env.step(action.item(), step_number)

        self.update_history(episode, info)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)

        return action, next_state, done, info

    def predict(self, state_string):
        """
        Predict the solution from the given state_string.
        """
        state = self.convert_state(state_string)
        state = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        done = False
        t = 0
        losses = []
        while not done:
            _, _, _, done = self.step(state, training=False)
            loss = self.env.find_loss(self.env.state_string)
            losses.append(loss)
            t += 1

            if self.is_constant_loss(losses):
                done = True

        logger.info(f"Solver terminated after {t} steps. Final "
                    f"state = {self.env.state_string} with loss = {loss}.")

    def is_constant_loss(self, losses):
        """Check for constant loss over a long number of steps"""
        check = (len(losses) >= Config.RESET_STEPS
                 and len(set(losses[-Config.RESET_STEPS:])) <= 1)
        if check:
            logger.info(f'Loss has been constant ({list(losses)[-1]}) '
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
