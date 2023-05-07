"""Example usage of hyperparameter tuner."""
from rex import init_logger

from rl_equation_solver.agent.a2c import Agent
from rl_equation_solver.utilities.tuner import Tuner

if __name__ == "__main__":
    init_logger(__name__, log_level="INFO")
    init_logger("rl_equation_solver", log_level="INFO")

    params = {
        "gamma": [0.9],
        "entropy_coef": [0.1, 0.2, 0.3],
        "critic_coef": [0.5, 0.6, 0.7, 0.9],
        "update_freq": [10],
        "learning_rate": [5e-4],
        "state_dim": [128],
    }

    env_kwargs = {"order": 2}

    tuner = Tuner(
        params=params,
        run_number=10,
        max_workers=32,
        env_kwargs=env_kwargs,
        agent=Agent,
    )

    tuner.run()
