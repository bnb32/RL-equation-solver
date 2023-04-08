"""Run quadratic solver: solve a*x^2 + b*x + c = 0"""
from rex import init_logger
from rl_equation_solver.environment.algebraic import AlgebraicEnv
from rl_equation_solver.agent.agent import Agent
from sympy import symbols

if __name__ == '__main__':
    init_logger(__name__, log_level='DEBUG')
    init_logger('rl_equation_solver', log_level='DEBUG')

    env = AlgebraicEnv(symbols('x a b c'))
    agent = Agent(env)
    agent.train(num_episodes=10)
    agent.predict(env._get_state())
