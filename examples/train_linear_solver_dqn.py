"""Run linear solver"""
from rex import init_logger
from rl_equation_solver.env_linear_equation import Env
from rl_equation_solver.q_learning.network import QLearningModel

if __name__ == '__main__':
    init_logger(__name__, log_level='DEBUG')
    init_logger('rl_equation_solver', log_level='DEBUG')

    env = Env()
    qmodel = QLearningModel(env)
    qmodel.train(num_episodes=10)
