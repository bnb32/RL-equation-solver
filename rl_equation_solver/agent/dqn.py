"""Agent with DQN based policy"""
from torch import optim
import logging

from rl_equation_solver.agent.base import BaseAgent, Config, ReplayMemory
from rl_equation_solver.agent.networks import DQN


logger = logging.getLogger(__name__)


class Agent(BaseAgent):
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
        super().__init__(env, hidden_size)
        n_actions = env.action_space.n
        n_observations = env.observation_space.n
        self.memory = ReplayMemory(Config.MEM_CAP)
        self.policy_network = DQN(n_observations, n_actions,
                                  hidden_size).to(self.device)
        self.target_network = DQN(n_observations, n_actions,
                                  hidden_size).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.AdamW(self.policy_network.parameters(),
                                     lr=Config.LR, amsgrad=True)

        logger.info(f'Initialized Agent with device {self.device}')

    def init_state(self):
        """Initialize state as a vector"""
        state_string = self.env._init_state()
        self.env.state_string = state_string
        self.env.state_vec = self.env.to_vec(state_string)
        return self.env.state_vec

    def convert_state(self, state):
        """Convert state string to vector representation"""
        return self.env.to_vec(state)
