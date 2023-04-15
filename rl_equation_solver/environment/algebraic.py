"""Environment for linear equation solver"""

from gym import spaces
from sympy import symbols
from operator import add, sub, truediv, pow
import logging

from rl_equation_solver.config import Config
from rl_equation_solver.utilities import utilities


logger = logging.getLogger(__name__)


class Env:
    """
    Environment for solving algebraic equations using RL.

    Example
    -------
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

    def __init__(self, order=2):
        """
        Parameters
        ----------
        order : int
            Order of alegraic equation. e.g. if order = 2 then the equation
            to solve will be a1 * x + a0 = 0
        """

        # Initialize the state
        self.order = order
        self.state_string = None
        self.operations = [add, sub, truediv, pow]
        self._actions = None
        self._terms = None
        self._feature_dict = None
        self._state_vec = None
        self._state_graph = None
        self._equation = None

        self.max_loss = 50
        self.state_dim = Config.VEC_DIM
        self.state = self._init_state()

        # Gym compatibility
        self.action_dim = len(self.actions)
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Discrete(self.state_dim)

    @property
    def feature_dict(self):
        """Get the feature dictionary"""
        if self._feature_dict is None:
            self._feature_dict = self._get_feature_dict()
        return self._feature_dict

    @property
    def terms(self):
        """Get list of fundamental terms"""
        if self._terms is None:
            self._terms = self._get_terms()
        return self._terms

    @property
    def actions(self):
        """Get list of fundamental actions"""
        if self._actions is None:
            self._actions = self._get_actions()
        return self._actions

    @property
    def equation(self):
        """Get equation from symbols"""
        if self._equation is None:
            self._equation = self._get_equation()
        return self._equation

    def reset(self):
        """
        Reset the environment state

        Returns
        -------
        state_vec : np.ndarray
            Vector representing initial state
        """
        _ = self._init_state()
        return self.state_vec

    def _get_symbols(self):
        """
        Get equation symbols. e.g. symbols('x a b')

        Returns
        -------
        symbols
        """
        symbol_list = 'x '
        symbol_list += ' '.join([f'a{i}' for i in range(self.order)[::-1]])
        return symbols(symbol_list)

    def _get_terms(self):
        """Get terms for quadratic equation"""
        _, *coeffs = self._get_symbols()
        return [*coeffs, 1]

    @property
    def state_vec(self):
        """Get current state vector"""
        self._state_vec = utilities.to_vec(self.state_string,
                                           self.feature_dict,
                                           self.state_dim)
        return self._state_vec

    @state_vec.setter
    def state_vec(self, value):
        """Set state_vec value"""
        self._state_vec = value

    @property
    def state_graph(self):
        """Get current state graph"""
        self._state_graph = utilities.to_graph(self.state_string,
                                               self.feature_dict)
        return self._state_graph

    @state_graph.setter
    def state_graph(self, value):
        """Set state_graph value"""
        self._state_graph = value

    @property
    def node_labels(self):
        """Get node labels for current state graph"""
        return utilities.get_node_labels(self.state_graph)

    def _init_state(self):
        """
        Get environment state

        Returns
        -------
        state_string : str
            State string representing environment state
        """
        *_, init = self._get_symbols()
        self.state_string = -init
        return self.state_string

    def _get_equation(self):
        """
        Simple linear equation

        Returns
        -------
        eqn : Object
            Equation object constructed from symbols
        """
        x, *coeffs, const = self._get_symbols()
        eqn = const
        for i, coeff in enumerate(coeffs[::-1]):
            eqn += coeff * pow(x, i + 1)
        return eqn

    def _get_actions(self):
        """
        Operations x terms

        Returns
        -------
        actions : list
            List of operation, term pairs
        """
        illegal_actions = [[truediv, 0]]
        actions = [[op, term] for op in self.operations for term in self.terms
                   if [op, term] not in illegal_actions]
        return actions

    def _get_feature_dict(self):
        """Return feature dict representing features at each node"""
        keys = ['Add', 'Mul', 'Pow']
        keys += [str(sym) for sym in self._get_symbols()]
        return {key: -(i + 2) for i, key in enumerate(keys)}

    # pylint: disable=unused-argument
    def render(self, mode='human'):
        """
        Print the state string representation
        """
        print(self.state)
