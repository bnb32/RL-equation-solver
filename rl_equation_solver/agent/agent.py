"""DQN module"""
import math
import random
from collections import namedtuple, deque
from itertools import count
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import logging


logger = logging.getLogger(__name__)


class Config:
    """Model configuration"""

    # BATCH_SIZE is the number of Experience sampled from the replay buffer
    BATCH_SIZE = 128
    # GAMMA is the discount factor
    GAMMA = 0.99
    # EPSILON_START is the starting value of epsilon
    EPSILON_START = 0.9
    # EPSILON_END is the final value of epsilon
    EPSILON_END = 0.05
    # EPSILON_DECAY controls the rate of exponential decay of epsilon, higher
    # means a slower decay
    EPSILON_DECAY = 1000
    # TAU is the update rate of the target network
    TAU = 0.005
    # LR is the learning rate of the AdamW optimizer
    LR = 1e-4
    # the hidden layers in the DQN
    HIDDEN_SIZE = 128
    # memory capacity
    MEM_CAP = 10000
    # reset after this many steps with constant loss
    RESET_STEPS = 100


# structure of the Experiences to store
Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """Stores the Experience Replay buffer"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def apply_action(self, *args):
        """Save the Experience into memory"""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """select a random batch of Experience for training"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """Simple MLP network."""

    def __init__(self, n_observations, n_actions, hidden_size):
        """
        Parameters
        ----------
        n_observations: int
            observation/state size of the environment
        n_actions : int
            number of discrete actions available in the environment
        hidden_size : int
            size of hidden layers
        """
        super().__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        """
        Forward pass for given state x
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Agent:
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
        n_actions = env.action_space.n
        n_observations = env.observation_space.n
        self.steps_done = 0
        self.memory = ReplayMemory(Config.MEM_CAP)
        self.device = torch.device('mps:0' if torch.backends.mps.is_available()
                                   else 'cpu')
        self.policy_network = DQN(n_observations, n_actions,
                                  hidden_size).to(self.device)
        self.target_network = DQN(n_observations, n_actions,
                                  hidden_size).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.AdamW(self.policy_network.parameters(),
                                     lr=Config.LR, amsgrad=True)

    @property
    def history(self):
        """Get training history"""
        return self.env.history

    def choose_optimal_action(self, state):
        """
        Choose action with max expected reward := max a * Q(s, a)

        max(1) will return largest column value of each row. second column on
        max result is index of where max element was found so we pick action
        with the larger expected reward.
        """
        with torch.no_grad():
            return self.policy_network(state).max(1)[1].view(1, 1)

    def choose_action(self, state):
        """
        Choose action based on given state. Either choose optimal action or
        random action depending on training step.
        """
        random_float = random.random()
        decay = (Config.EPSILON_START - Config.EPSILON_END)
        decay *= math.exp(-1. * self.steps_done / Config.EPSILON_DECAY)
        epsilon_threshold = Config.EPSILON_END + decay

        self.steps_done += 1
        if random_float > epsilon_threshold:
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
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            total_reward = 0
            for t in count():
                # sample an action
                action = self.choose_action(state)
                # execute it, observe the next screen and the reward
                observation, reward, done, _ = self.env.step(action.item(),
                                                             training=True)
                reward = torch.tensor([reward], device=self.device)

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32,
                                              device=self.device).unsqueeze(0)

                # Store the experience in the memory
                self.memory.apply_action(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Kick agent out of local minima
                losses = self.history['loss'][-Config.RESET_STEPS:]
                if len(losses) >= Config.RESET_STEPS and len(set(losses)) <= 1:
                    logger.info(f'Loss has been constant ({list(losses)[0]}) '
                                f'for {Config.RESET_STEPS} steps. Reseting.')
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
                total_reward += reward
                if done:
                    episode_duration.append(t + 1)
                    logger.info(f"Episode {i}, Solver terminated after {t} "
                                f"steps with reward {total_reward}. Final "
                                f"state = {self.env.state_string}")
                    break

    def predict(self, state_string):
        """
        Predict the solution from the given state_string.
        """
        state = self.env.to_vec(state_string)
        state = torch.tensor(state, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        done = False
        t = 0
        while not done:
            action = self.choose_optimal_action(state)
            _, _, done, _ = self.env.step(action.item())
            loss = self.env.find_loss(self.env.state_string)
            t += 1

        logger.info(f"Solver terminated after {t} steps. Final "
                    f"state = {self.env.state_string} with loss = {loss}.")

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
