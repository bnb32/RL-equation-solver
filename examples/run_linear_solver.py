"""Run linear solver"""
from rex import init_logger
from sympy import symbols
from rl_equation_solver.environment.linear import Env
from rl_equation_solver.agent.agent import Agent

if __name__ == '__main__':
    init_logger(__name__, log_level='DEBUG')
    init_logger('rl_equation_solver', log_level='DEBUG')

    env = Env()
    agent = Agent(env)
    agent.train(num_episodes=10)
    agent.predict(env._get_state())
