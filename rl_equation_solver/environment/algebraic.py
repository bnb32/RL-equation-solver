"""Environment for linear equation solver"""
import logging
from operator import add, pow, sub, truediv

import gym
import numpy as np
from gym import spaces
from sympy import expand, nsimplify, parse_expr, simplify, symbols

from rl_equation_solver.config import DefaultConfig
from rl_equation_solver.utilities import utilities
from rl_equation_solver.utilities.history import History
from rl_equation_solver.utilities.reward import RewardMixin

logger = logging.getLogger(__name__)


class Env(gym.Env, RewardMixin, History):
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

        History.__init__(self)

        # Initialize the state
        self.order = order
        self._state_string = None
        self._next_state_string = None
        self._operations = None
        self._actions = None
        self._terms = None
        self._feature_dict = None
        self._state_vec = None
        self._next_state_vec = None
        self._state_graph = None
        self._equation = None
        self.window = None
        self.config = config
        self.reward_function = None
        self.state_dim = None
        self._initial_state = init_state
        self.state_string = self.initial_state

        self.init_config()

        # Gym compatibility
        self.action_dim = len(self.actions)
        self.action_space = spaces.Discrete(self.action_dim)
        min_val = min(self.feature_dict.values())
        self.observation_space = spaces.Box(
            min_val,
            min_val + self.state_dim,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.n_actions = self.action_space.n
        self.n_obs = self.observation_space.shape[0]

        logger.info(
            f"Initializing environment with order={order}, |S| = "
            f"{self.n_actions} x {self.n_obs} = "
            f"{self.n_actions * self.n_obs}"
        )
        logger.info(f"Using reward function: {self.reward_function}.")

    def init_config(self):
        """Initialize model configuration"""
        config = DefaultConfig
        if self.config is not None:
            config.update(self.config)
        for key, val in config.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def update_config(self, config: dict) -> None:
        """Update configuration"""
        self.config = config
        self.init_config()

    @property
    def state_string(self):
        """Get string representation of the solution state"""
        return self._state_string

    @state_string.setter
    def state_string(self, value):
        """Set string representation of solution state"""
        self._state_string = value

    @property
    def next_state_string(self):
        """Get string representation of the next solution state"""
        return self._next_state_string

    @next_state_string.setter
    def next_state_string(self, value):
        """Set string representation of next solution state"""
        self._next_state_string = value

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

    def _get_algebraic_symbols(self):
        """
        Get equation symbols. e.g. symbols('x a b')

        Returns
        -------
        symbols
        """
        symbol_list = "x "
        symbol_list += " ".join([f"a{i}" for i in range(self.order)])
        return symbols(symbol_list)

    def _get_numerical_symbols(self):
        """Get numerical symbols like '0', '1', '-1'"""
        return symbols("0 1")

    def _get_terms(self):
        """Get terms for quadratic equation"""
        _, *coeffs = self._get_algebraic_symbols()
        terms = [*coeffs, *self._get_numerical_symbols()]
        for n in range(2, self.order):
            terms.append(symbols(f"1 / {n}"))
        return terms

    @property
    def state_vec(self):
        """Get current state vector"""
        self._state_vec = utilities.to_vec(
            self.state_string, self.feature_dict, self.state_dim
        )
        return self._state_vec

    @state_vec.setter
    def state_vec(self, value):
        """Set state_vec value"""
        self._state_vec = value

    @property
    def next_state_vec(self):
        """Get next state vector"""
        self._next_state_vec = utilities.to_vec(
            self.next_state_string, self.feature_dict, self.state_dim
        )
        return self._next_state_vec

    @property
    def state_graph(self):
        """Get current state graph"""
        self._state_graph = utilities.to_graph(self.state_string, self.feature_dict)
        return self._state_graph

    @state_graph.setter
    def state_graph(self, value):
        """Set state_graph value"""
        self._state_graph = value

    @property
    def node_labels(self):
        """Get node labels for current state graph"""
        return utilities.get_node_labels(self.state_graph)

    @property
    def initial_state(self):
        """
        Initialize environment state
        """
        if self._initial_state is None:
            self._initial_state = symbols("0")
        return self._initial_state

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
        self.loop_step = 0
        self.state_string = self.initial_state
        return self.state_vec

    def _get_equation(self):
        """
        Simple linear equation

        Returns
        -------
        eqn : Object
            Equation object constructed from symbols
        """
        x, *coeffs, const = self._get_algebraic_symbols()
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
        illegal_actions = [
            [truediv, symbols("0")],
            [add, symbols("0")],
            [sub, symbols("0")],
            [pow, symbols("1")],
            [pow, symbols("0")],
        ]
        actions = [
            [op, term]
            for op in self.operations
            for term in self.terms
            if [op, term] not in illegal_actions
        ]
        return actions

    def _get_feature_dict(self):
        """Return feature dict representing features at each node"""
        keys = ["Add", "Mul", "Pow"]
        keys += [str(sym) for sym in self._get_algebraic_symbols()]
        keys += [str(sym) for sym in self._get_numerical_symbols()]
        keys += ["I"]
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
        if not hasattr(self, self.reward_function):
            raise ValueError(
                "Env has no reward function named " f"{self.reward_function}"
            )
        method = getattr(self, self.reward_function)
        reward = method(state_old, state_new)
        return reward

    def extra_reward(self, reward):
        """Extra penalty / reward for time elapsed, if solution found and if
        state too long"""
        if self.too_long(self.next_state_vec):
            reward -= 100
        if self.complexity == 0:
            reward += 100 / (self.loop_step + 1)
        return reward

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

    def get_expr_number_count(self, expr):
        """Count all numbers in the given expression. This is to increase the
        compexity of an expression if it includes higher numbers. e.g. 4*a
        should have higher complexity than 2*a"""

        numbers = expr.find(lambda e: e.is_Number, group=True)
        n_count = 0
        for n in numbers:
            if "/" in str(n):
                a, b = str(n).split("/")
                n_count += abs(float(a))
                n_count += abs(float(b))
            else:
                n_count += abs(float(n))
        return n_count

    def get_expression_complexity(self, expr):
        """
        Compute graph / expression complexity for the given expression

        Parameters
        ----------
        expr : str
            String representation of the expression

        Returns
        -------
        complexity : int
            Number of edges plus number of nodes in graph representation /
            expression_tree of given expression
        """
        if parse_expr(str(expr)) == 0:
            complexity = 0
        else:
            state_graph = utilities.to_graph(expr, self.feature_dict)
            complexity = state_graph.number_of_nodes()
            complexity += state_graph.number_of_edges()
            complexity += self.get_expr_number_count(expr)

        return complexity

    def get_solution_complexity(self, state):
        """Get the graph / expression complexity for a given state. This is
        equal to number_of_nodes + number_of_edges"""
        soln = self.get_solution_approx(state)
        return self.get_expression_complexity(soln)

    def get_solution_approx(self, state):
        """Get the approximate solution from the given state"""
        expr = self.equation.replace(symbols("x"), nsimplify(state))
        solution_approx = simplify(expand(expr))
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
        self.next_state_string = self.get_next_state(action)

        self.reward = self.find_reward(self.state_string, self.next_state_string)

        self.reset_state_check(self.next_state_vec)

        self.solution_approx = self.get_solution_approx(self.next_state_string)
        self.complexity = self.get_expression_complexity(self.solution_approx)

        self.reward = self.extra_reward(self.reward)

        self.append_history(self.info)

        msg = self.get_log_info()
        logger.debug(msg)

        done = self.check_if_done()

        self.state_string = self.next_state_string

        return self.state_vec, self.reward, done, self.info

    def check_if_done(self):
        """Check if solution was found or max steps was reached"""
        done = self.max_steps_reached() or self.complexity == 0
        if done:
            self.current_episode += 1
            msg = self.get_log_info()
            logger.info(msg)
        self.steps_done += 1
        self.loop_step += 1

        return done

    def max_steps_reached(self):
        """Check if max steps was reached"""
        check = self.loop_step > self.max_solution_steps
        if check:
            logger.info(
                f"loop_step {self.loop_step} exceeded max " f"{self.max_solution_steps}"
            )
        return check

    def reset_state_check(self, state_vec: np.ndarray):
        """Check if state is too long and reset state if True"""
        self.reset_step = self.too_long(state_vec)
        if self.reset_step:
            self.next_state_string = self.initial_state

    def get_next_state(self, action: int):
        """Get next state from given action"""
        [operation, term] = self.actions[action]
        return operation(self.state_string, term)

    # pylint: disable=unused-argument
    def render(self, mode: str = "human"):
        """
        Print the state string representation
        """
        print(self.state_string)
