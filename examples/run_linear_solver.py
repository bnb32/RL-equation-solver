"""Run linear solver"""
from rex import init_logger
from rl_equation_solver.environment.linear_equation import Env

if __name__ == '__main__':
    init_logger(__name__, log_level='DEBUG')
    init_logger('rl_equation_solver', log_level='DEBUG')

    Env.run()
