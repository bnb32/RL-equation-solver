"""Environment for linear equation solver"""

from sympy import symbols
import logging

from rl_equation_solver.environment.base import BaseEnv


logger = logging.getLogger(__name__)


class Env(BaseEnv):
    """
    Env for solving algebraic equations using RL. Warm up with simple
    equations

    a x + b = 0

    The agent starts at state = 1 and chooses
    an action by combing operations and terms:

    operations: (add, subtract, mulitple, divide, pow)
    terms: (a, b, 0, 1)

    action[i][j] = (operation[i], terms[j])

    So taking action[0][0] = (add, a) in state 1 would result in

    new_state = a + 1

    Followed by an action (div, b) would result in

    new_state = (a + 1) / b

    The states are represented using sympy and can be mapped onto a directed
    acyclic graph (dag). These state representation is what we will feed the
    RL agent.

    The agent is rewarded if it reduces the "loss" of the equation defined as
    the length of the state graph -- intuitively, the complexity of the state:

    loss = num_nodes + num_leaves of state graph

    If the agent finds the solution, the equation terminates.
    """

    metadata = {"render.modes": ["human"]}

    def _get_symbols(self):
        """Get symbols for linear equation"""
        return symbols('x a b')

    def _get_terms(self):
        """Get terms for linear equation"""
        _, a, b = self._get_symbols()
        return [a, b, 0, 1]

    def _get_equation(self):
        """
        Simple linear equation

        Returns
        -------
        eqn : Object
            Equation object constructed from symbols
        """
        x, a, b, = self._get_symbols()
        eqn = a * x + b
        return eqn

    def _get_feature_dict(self):
        """Return feature dict representing features at each node"""
        keys = ['Add', 'Mul', 'Pow'] + ['x', 'a', 'b']
        return {key: -(i + 2) for i, key in enumerate(keys)}
