"""
Run linear solver with DQN based agent: solve :math:`ax + b = 0`
"""
from rex import init_logger
from rl_equation_solver.environment.algebraic import Env
from rl_equation_solver.agent.dqn import Agent

if __name__ == '__main__':
    init_logger(__name__, log_level='DEBUG')
    init_logger('rl_equation_solver', log_level='DEBUG')

    env = Env(order=2)
    agent = Agent(env)
    agent.train(num_episodes=10)
    agent.predict(env._init_state())