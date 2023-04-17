"""Environment for linear equation solver"""
import gym
from gym import spaces
from sympy import symbols, nsimplify, simplify, parse_expr
from operator import add, sub, truediv, pow
import logging
import numpy as np

from rl_equation_solver.config import DefaultConfig
from rl_equation_solver.utilities import utilities
from rl_equation_solver.utilities.reward import RewardMixin
from rl_equation_solver.utilities.history import HistoryMixin


logger = logging.getLogger(__name__)


class Env(gym.Env, RewardMixin, HistoryMixin):
    r"""
    Environment for solving algebraic equations using RL.

    Example
    -------
    :math:`a x + b = 0`

    The agent starts at state = 1 and chooses
    an action by combing operations and terms:

    operations: (add, subtract, mulitple, divide, pow)
    terms: (a, b, 0, 1)

    action[i][j] = (operation[i], terms[j])

    So taking action[0][0] = (add, a) in state 1 would result in

    new_state = :math:`a + 1`

    Followed by an action (div, b) would result in

    new_state = :math:`(a + 1) / b`

    The states are represented using sympy and can be mapped onto a directed
    acyclic graph (dag). These state representation is what we will feed the
    RL agent.

    The agent is rewarded if it reduces the "loss" of the equation defined as
    the length of the state graph -- intuitively, the complexity of the state:

    loss = num_nodes + num_leaves of state graph

    If the agent finds the solution, the equation terminates.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, order=2, init_state=None, config=None):
        """
        Parameters
        ----------
        order : int
            Order of alegraic equation. e.g. if order = 2 then the equation
            to solve will be a0 * x + a1 = 0
        init_state : sympy.Equation | None
            Optional initial guess for equation solution. e.g. -b/a, using
            symbols from sympy.symbols('x a b'). If None then initial guess
            will be (-1) * constant_term.
        config : dict | None
            Model configuration. If None then the default model configuration
            in rl_equation_solver.config will be used.
        """

        HistoryMixin.__init__(self)

        # Initialize the state
        self.order = order
        self._state_string = None
        self._operations = None
        self._actions = None
        self._terms = None
        self._feature_dict = None
        self._state_vec = None
        self._state_graph = None
        self._equation = None
        self.info = None
        self.loop_step = 0
        self.steps_done = 0
        self.current_episode = 0
        self.window = None
        self.config = config

        self.state_dim = None
        self._initial_state = init_state

        self.init_config()

        self.state_string = init_state or self._init_state()

        # Gym compatibility
        self.action_dim = len(self.actions)
        self.action_space = spaces.Discrete(self.action_dim)
        min_val = min(self.feature_dict.values())
        self.observation_space = spaces.Box(min_val,
                                            min_val + self.state_dim,
                                            shape=(self.state_dim,),
                                            dtype=np.float32)
        self.n_actions = self.action_space.n
        self.n_obs = self.observation_space.shape[0]

        logger.info(f'Initializing environment with order={order}, |S| = '
                    f'{self.n_actions} x {self.n_obs} = '
                    f'{self.n_actions * self.n_obs}')

    def init_config(self):
        """Initialize model configuration"""
        config = DefaultConfig
        if self.config is not None:
            config.update(self.config)
        for key, val in config.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def update_config(self, config):
        """Update configuration"""
        self.config = config
        self.init_config()

    @property
    def state_string(self):
        """Get string representation of the solution state"""
        return nsimplify(parse_expr(str(self._state_string)))

    @state_string.setter
    def state_string(self, value):
        """Set string representation of solution state"""
        self._state_string = value

    @property
    def operations(self):
        """Get list of valid operations"""
        if self._operations is None:
            self._operations = [add, sub, truediv, pow]
        return self._operations

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

    def _get_symbols(self):
        """
        Get equation symbols. e.g. symbols('x a b')

        Returns
        -------
        symbols
        """
        symbol_list = 'x '
        symbol_list += ' '.join([f'a{i}' for i in range(self.order)])
        return symbols(symbol_list)

    def _get_terms(self):
        """Get terms for quadratic equation"""
        _, *coeffs = self._get_symbols()
        terms = [*coeffs, 0, 1]
        for n in range(2, self.order):
            terms.append(1 / n)
        return terms

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
        Initialize environment state
        """
        self.loop_step = 0
        if self._initial_state is None:
            self._initial_state = symbols('1')
        self.state_string = self._initial_state

    # pylint: disable=unused-argument
    def reset(self, seed=None, options=None):
        """
        Reset environment state

        Returns
        -------
        state_vec : np.ndarray
            State vector representing environment state
        info : dict
            Dictionary with training info
        """
        self._init_state()
        return self.state_vec

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
        illegal_actions = [[truediv, 0], [add, 0], [sub, 0], [pow, 1],
                           [pow, 0]]
        actions = [[op, term] for op in self.operations for term in self.terms
                   if [op, term] not in illegal_actions]
        return actions

    def _get_feature_dict(self):
        """Return feature dict representing features at each node"""
        keys = ['Add', 'Mul', 'Pow']
        keys += [str(sym) for sym in self._get_symbols()]
        keys += ['I']
        return {key: -(i + 2) for i, key in enumerate(keys)}

    def find_reward(self, state_old, state_new):
        """
        Parameters
        ----------
        state_old : str
            String representation of last state
        state_new : str
            String representation of new state

        Returns
        -------
        reward : int
            Difference between loss for state_new and state_old
        """
        return self.diff_loss_reward(state_old, state_new)

    def too_long(self, state):
        """
        Check if state dimension is too large

        Parameters
        ----------
        state : str
            State string representation

        Returns
        -------
        bool
        """
        return len(state) > self.state_dim

    def expression_complexity(self, state):
        """
        Compute graph / expression complexity for the given state

        Parameters
        ----------
        state : str
            String representation of the current state

        Returns
        -------
        complexity : int
            Number of edges plus number of nodes in graph representation /
            expression_tree of the current solution approximation
        """
        solution_approx = self._get_solution_approx(state)
        if solution_approx == 0:
            complexity = 0
        else:
            state_graph = utilities.to_graph(solution_approx,
                                             self.feature_dict)
            complexity = state_graph.number_of_nodes()
            complexity += state_graph.number_of_edges()

        return complexity

    def _get_solution_approx(self, state):
        """Get the approximate solution from the given state"""
        replaced = self.equation.replace(symbols('x'),
                                         nsimplify(parse_expr(str(state))))
        solution_approx = simplify(replaced)
        return solution_approx

    def step(self, action: int):
        """
        Take step corresponding to the given action

        Parameters
        ----------
        action : int
            Action index corresponding to the entry in the action list
            constructed in _make_physical_actions
        step_number : int
            Number of steps taken so far.

        Returns
        -------
        new_state : Tensor | GraphEmbedding
            New state after action. Represented as a pytorch Tensor or
            GraphEmbedding
        reward : float
            Reward from taking this step
        done : bool
            Whether problem is solved or if maximum state dimension is reached
        info : dict
            Additional information
        """
        # action is 0,1,2,3, ...,  get the physical actions it indexes
        [operation, term] = self.actions[action]
        new_state_string = operation(self.state_string, term)
        new_state_string = nsimplify(new_state_string)
        new_state_vec = utilities.to_vec(new_state_string,
                                         self.feature_dict,
                                         self.state_dim)

        # Reward
        reward = self.find_reward(self.state_string, new_state_string)

        # Done
        done = False
        if self.too_long(new_state_vec):
            done = True

        # If complexity is zero, you have solved the problem
        complexity = self.expression_complexity(new_state_string)
        if complexity == 0:
            done = True

        # Update
        if not done or complexity == 0:
            self.state_string = new_state_string

        if complexity == 0:
            logger.info(f'solution is: {self.state_string}')

            # reward finding solution in fewer steps
            reward += 10 / (1 + self.loop_step)

        # Extra info
        self.info = {'ep': self.current_episode,
                     'step': self.steps_done,
                     'complexity': complexity,
                     'loss': np.nan,
                     'reward': reward,
                     'state': self.state_string}
        self.append_history(self.info)

        self.steps_done += 1
        self.loop_step += 1

        if done:
            self.log_info()
            self.current_episode += 1

        return self.state_vec, reward, done, self.info

    # pylint: disable=unused-argument
    def render(self, mode='human'):
        """
        Print the state string representation
        """
        print(self.state_string)
