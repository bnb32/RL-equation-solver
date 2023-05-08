"""A2C implementation."""
from typing import Optional

from rl_equation_solver.agent.networks import GraphActorCritic
from rl_equation_solver.agent.on_policy import OnPolicyAgent
from rl_equation_solver.agent.state import GraphState
from rl_equation_solver.environment.algebraic import Env


class Agent(GraphState, OnPolicyAgent):
    """A2C Agent."""

    def __init__(
        self, env: Env, config: Optional[dict] = None, **kwargs
    ) -> None:
        """Initialize A2C Agent."""
        OnPolicyAgent.__init__(self, env, config, **kwargs)
        GraphState.__init__(
            self,
            env=self.env,
            n_observations=self.n_observations,
            n_actions=self.n_actions,
            feature_num=self.feature_num,
        )
        self.model = GraphActorCritic(
            self.n_observations, self.n_actions, self.hidden_size
        ).to(self.device)

        self.init_optimizer()
