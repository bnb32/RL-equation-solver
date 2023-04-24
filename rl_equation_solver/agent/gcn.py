"""Agent with GCN based policy"""
import logging

from rl_equation_solver.agent.base import OffPolicyAgent, GraphState
from rl_equation_solver.agent.networks import GCN
from rl_equation_solver.utilities import utilities
from rl_equation_solver.utilities.utilities import GraphEmbedding

logger = logging.getLogger(__name__)


class Agent(GraphState, OffPolicyAgent):
    """Agent with GCN target and policy networks"""

    def __init__(self, policy="MlpPolicy", env=None, config=None):
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
        device : str
            Device to use for torch objects. e.g. 'cpu' or 'cuda:0'
        """
        OffPolicyAgent.__init__(self, policy, env, config)
        GraphState.__init__(self, env=self.env,
                            n_observations=self.n_observations,
                            n_actions=self.n_actions,
                            feature_num=self.feature_num,
                            device=self.device)
        self.policy.policy_network = GCN(
            self.n_observations, self.n_actions, self.hidden_size
        ).to(self.device)
        self.policy.target_network = GCN(
            self.n_observations, self.n_actions, self.hidden_size
        ).to(self.device)
        self.policy.target_network.load_state_dict(
            self.policy.policy_network.state_dict()
        )
        self.policy.init_optimizer()

        logger.info(f"Initialized Agent with device {self.device}")
