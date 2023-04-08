"""Agent with GCN based policy"""
from torch import optim
import logging

from rl_equation_solver.agent.base import BaseAgent, Config, ReplayMemory
from rl_equation_solver.agent.networks import GCN


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
        self.env = env
        n_actions = env.action_space.n
        n_observations = env.observation_space.n
        self.steps_done = 0
        self.memory = ReplayMemory(Config.MEM_CAP)
        self.policy_network = GCN(n_observations, n_actions,
                                  hidden_size).to(self.device)
        self.target_network = GCN(n_observations, n_actions,
                                  hidden_size).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = optim.AdamW(self.policy_network.parameters(),
                                     lr=Config.LR, amsgrad=True)

        logger.info(f'Initialized Agent with device {self.device}')

    def init_state(self):
        """Initialize state as a graph"""
        state_string = self.env._get_state()
        self.env.state_string = state_string
        self.env.graph, _ = self.env.to_graph(state_string)
        return self.env.graph

    def convert_state(self, state):
        """Convert state string to graph representation"""
        return self.env.to_graph(state)
