"""Example usage of hyperparameter tuner."""
from rex import init_logger

from rl_equation_solver.agent.a2c import Agent
from rl_equation_solver.utilities.tuner import Tuner

if __name__ == "__main__":
    init_logger(__name__, log_level="INFO")
    init_logger("rl_equation_solver", log_level="INFO")

    params = {
        "gamma": [0.7, 0.8, 0.9],
        "entropy_coef": [0.2, 0.25, 0.3, 0.35],
        "critic_coef": [0.7, 0.75, 0.8, 0.85, 0.9],
        "update_freq": [10],
        "state_dim": [128],
        "learning_rate": [5e-4, 1e-4, 5e-3],
    }

    env_kwargs = {"order": 2}

    tuner = Tuner(
        params=params,
        run_number=15,
        max_workers=16,
        env_kwargs=env_kwargs,
        agent=Agent,
    )

    best_config = tuner.run(optimize_key="step", get_min=True)
