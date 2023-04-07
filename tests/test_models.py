"""Test model loading, training, and running"""
from rl_equation_solver.environment.linear import Env
from rl_equation_solver.agent.agent import Agent

from rex import init_logger
from tempfile import TemporaryDirectory
import os
import pickle
import pprint


def test_model_load():
    """Test environment loading"""
    _ = Env()


def test_model_train(log=True):
    """Test DQN training"""
    if log:
        init_logger(__name__, log_level='DEBUG')
        init_logger('rl_equation_solver', log_level='DEBUG')

    env = Env()
    agent = Agent(env)
    agent.train(num_episodes=2)

    with TemporaryDirectory() as td:
        history_file = os.path.join(td, "history.pkl")
        with open(history_file, "wb") as fp:
            pickle.dump(agent.history, fp)
            print(pprint.pformat(agent.history, indent=4))


def test_model_predict(log=True):
    """Test Agent prediction"""

    if log:
        init_logger(__name__, log_level='DEBUG')
        init_logger('rl_equation_solver', log_level='DEBUG')

    env = Env()
    agent = Agent(env)
    agent.train(num_episodes=2)
    _, _, b = env._get_symbols()
    agent.predict(-b)


def test_model_save_load(log=True):
    """Test saving and loading of agent"""

    if log:
        init_logger(__name__, log_level='DEBUG')
        init_logger('rl_equation_solver', log_level='DEBUG')

    with TemporaryDirectory() as td:
        outfile = os.path.join(td, 'model_file.pkl')
        env = Env()
        agent = Agent(env)
        agent.train(num_episodes=2)
        agent.save(outfile)

        agent = Agent.load(env, outfile)
        _, _, b = env._get_symbols()
        agent.predict(-b)
