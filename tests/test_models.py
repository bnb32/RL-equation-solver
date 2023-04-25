"""Test model loading, training, and running"""
import os
import pickle
import pprint
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pytest
from rex import init_logger

from rl_equation_solver.agent.a2c import Agent as AgentA2C
from rl_equation_solver.agent.dqn import Agent as AgentDQN
from rl_equation_solver.agent.gcn import Agent as AgentGCN
from rl_equation_solver.agent.lstm import Agent as AgentLSTM
from rl_equation_solver.environment.algebraic import Env

Agents = [AgentLSTM, AgentA2C, AgentGCN, AgentDQN]
agent_names = ["lstm", "a2c", "gcn", "dqn"]
config = {"max_solution_steps": 10, "device": "cpu"}


def test_model_load():
    """Test environment loading"""
    _ = Env()


@pytest.mark.parametrize("i", list(range(len(Agents))))
def test_model_train(i, log=True, plot=True):
    """Test DQN training"""
    agent = Agents[i]
    agent_name = agent_names[i]
    if log:
        init_logger(__name__, log_level="DEBUG")
        init_logger("rl_equation_solver", log_level="DEBUG")

    env = Env()
    agent = agent(env=env, config=config)
    agent.train(num_episodes=2)

    with TemporaryDirectory() as td:
        history_file = os.path.join(td, "history.pkl")
        with open(history_file, "wb") as fp:
            pickle.dump(agent.history, fp)
            print(pprint.pformat(agent.history, indent=4))

    histories = {
        "best": env.best_history,
        "avg": env.avg_history,
        "end": env.end_history,
        "total": env.total_history,
    }

    if plot:
        for hist_name, hist in histories.items():
            fig, ax = plt.subplots(1, 3)
            x = list(range(len(hist["loss"])))
            ax[0].scatter(x, hist["complexity"])
            ax[0].set_title("complexity")
            ax[1].scatter(x, hist["reward"])
            ax[1].set_title("reward")
            ax[2].scatter(x, hist["loop_step"])
            ax[2].set_title("steps")

            with TemporaryDirectory() as td:
                fig_name = f"{td}/{agent_name}_history_{hist_name}.png"
                fig.savefig(fig_name)


@pytest.mark.parametrize("agent", Agents)
def test_model_predict(agent, log=True):
    """Test Agent prediction"""

    if log:
        init_logger(__name__, log_level="DEBUG")
        init_logger("rl_equation_solver", log_level="DEBUG")

    env = Env()
    agent = agent(env=env, config=config)
    agent.train(num_episodes=2)
    agent.predict(env._initial_state)


@pytest.mark.parametrize("agent", Agents)
def test_model_save_load(agent, log=True):
    """Test saving and loading of agent"""

    if log:
        init_logger(__name__, log_level="DEBUG")
        init_logger("rl_equation_solver", log_level="DEBUG")

    with TemporaryDirectory() as td:
        outfile = os.path.join(td, "model_file.pkl")
        env = Env()
        agent = agent(env=env, config=config)
        agent.train(num_episodes=2)
        agent.save(outfile)

        agent = agent.load(outfile)
        agent.predict(env._initial_state)
