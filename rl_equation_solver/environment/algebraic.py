"""Environment for linear equation solver"""
import gym
import pygame
from gym import spaces
from sympy import symbols, nsimplify, simplify, expand
from operator import add, sub, truediv, pow
import logging
import numpy as np

from rl_equation_solver.config import Config
from rl_equation_solver.utilities import utilities
from rl_equation_solver.utilities.reward import RewardMixin


logger = logging.getLogger(__name__)


class Env(gym.Env, RewardMixin):
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

    def __init__(self, order=2, init_state=None):
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
        """

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
        self._history = {}
        self.info = None
        self.loop_step_number = 0
        self.steps_done = 0
        self.episode_number = 0
        self.window = None

        self.max_loss = 50
        self.state_dim = Config.VEC_DIM
        self._initial_state = init_state
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

    @property
    def state_string(self):
        """Get string representation of the solution state"""
        return nsimplify(self._state_string)

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
        terms = [*coeffs, 0]
        for n in range(1, self.order):
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
        self.loop_step_number = 0
        if self._initial_state is None:
            *_, init = self._get_symbols()
            self._initial_state = -init
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
        illegal_actions = [[truediv, 0]]
        actions = [[op, term] for op in self.operations for term in self.terms
                   if [op, term] not in illegal_actions]
        return actions

    def _get_feature_dict(self):
        """Return feature dict representing features at each node"""
        keys = ['Add', 'Mul', 'Pow']
        keys += [str(sym) for sym in self._get_symbols()]
        keys += ['I']
        return {key: -(i + 2) for i, key in enumerate(keys)}

    @property
    def history(self):
        """Get training history of policy_network"""
        return self._history

    @history.setter
    def history(self, value):
        """Set training history of policy_network"""
        self._history = value

    def append_history(self, episode, entry):
        """Append latest step for training history of policy_network"""
        if episode not in self._history:
            self._history[episode] = {'complexity': [], 'loss': [],
                                      'reward': [], 'state': []}
        self._history[episode]['complexity'].append(entry['complexity'])
        self._history[episode]['loss'].append(entry.get('loss', np.nan))
        self._history[episode]['reward'].append(entry['reward'])
        self._history[episode]['state'].append(entry['state'])

    def update_history(self, episode, key, value):
        """Update latest step for training history of policy_network"""
        self._history[episode][key][-1] = value

    def log_info(self):
        """Write info to logger"""
        out = self.info.copy()
        out['reward'] = '{:.3e}'.format(out['reward'])
        out['loss'] = '{:.3e}'.format(out['loss'])
        logger.info(out)

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
        x, *_ = self._get_symbols()
        replaced = self.equation.replace(x, state)
        solution_approx = simplify(expand(nsimplify(replaced)))
        if solution_approx == 0:
            complexity = 0
        else:
            state_graph = utilities.to_graph(solution_approx,
                                             self.feature_dict)
            complexity = state_graph.number_of_nodes()
            complexity += state_graph.number_of_edges()

        return complexity

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
        new_state_string = simplify(new_state_string)
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
            reward += 10 / (1 + self.loop_step_number)

        # Extra info
        self.info = {'episode_number': self.episode_number,
                     'step_number': self.steps_done,
                     'complexity': complexity, 'loss': np.nan,
                     'reward': reward,
                     'state': nsimplify(self.state_string)}
        self.steps_done += 1
        self.loop_step_number += 1
        if done:
            self.log_info()

        self.append_history(self.episode_number, self.info)

        if done:
            self.episode_number += 1

        return self.state_vec, reward, done, self.info

    # pylint: disable=unused-argument
    def render(self, mode='human'):
        """
        Print the state string representation
        """
        print(self.state_string)

    def close(self):
        """Close resources"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
