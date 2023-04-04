"""Test model loading, training, and running"""
from rl_equation_solver.env_linear_equation import Env
from rl_equation_solver.q_learning.network import QLearningModel


def test_model_load():
    """Test environment loading"""
    _ = Env()


def test_model_run():
    """Test environment run"""
    env = Env()
    env.run()


def test_model_train():
    """Test DQN training"""
    env = Env()
    qmodel = QLearningModel(env)
    qmodel.train(num_episodes=2)
