"""Agent with GCN based policy"""
import logging

from rl_equation_solver.agent.base import BaseAgent, Config, ReplayMemory
from rl_equation_solver.agent.networks import GCN
from rl_equation_solver.utilities import utilities
from rl_equation_solver.utilities.utilities import GraphEmbedding


logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """Agent with GCN target and policy networks"""

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
        super().__init__(env, hidden_size, device=device)
        self.memory = ReplayMemory(Config.MEM_CAP)
        self.policy_network = GCN(self.n_observations, self.n_actions,
                                  hidden_size).to(self.device)
        self.target_network = GCN(self.n_observations, self.n_actions,
                                  hidden_size).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.init_optimizer()

        logger.info(f'Initialized Agent with device {self.device}')

    def init_state(self):
        """Initialize state as a graph"""
        self.env._init_state()
        self.env.graph = utilities.to_graph(self.env.state_string,
                                            self.env.feature_dict)
        return GraphEmbedding(self.env.graph,
                              n_observations=self.n_observations,
                              n_features=Config.FEATURE_NUM,
                              device=self.device)

    def convert_state(self, state):
        """Convert state string to graph representation"""
        self.env.graph = utilities.to_graph(state,
                                            self.env.feature_dict)
        return GraphEmbedding(self.env.graph,
                              n_observations=self.n_observations,
                              n_features=Config.FEATURE_NUM,
                              device=self.device)

    def batch_states(self, states, device):
        """Batch agent states"""
        batch = utilities.Batch()(states, device)
        return batch
