"""Environment for linear equation solver"""

import gym
from gym import spaces
import numpy as np
from sympy import simplify
from operator import add, sub, mul, truediv, pow
import logging
from abc import abstractmethod

import networkx as nx
from networkx.readwrite import json_graph
from networkx.drawing.nx_pydot import graphviz_layout


logger = logging.getLogger(__name__)


class Id:
    """A helper class for autoincrementing node numbers."""
    counter = 0

    @classmethod
    def get(cls):
        """
        Get the node number
        """
        cls.counter += 1
        return cls.counter


class Node:
    """Represents a single operation or atomic argument."""

    def __init__(self, label, expr_id):
        self.id = expr_id
        self.name = label

    def __repr__(self):
        return self.name


class BaseEnv(gym.Env):
    """
    Base environment for solving algebraic equations using RL.

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

    def __init__(self):

        # Initialize the state
        self.state_string = None
        self.state_vec = None
        self.feature_dict = {}

        self.max_loss = 50
        self.state_dim = 1024
        self.equation = self._get_equation()
        self.actions = self._make_physical_actions()
        self.state = self._get_state()

        # Gym compatibility
        self.action_dim = len(self.actions)
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Discrete(self.state_dim)

    def reset(self):
        """
        Reset the environment state

        Returns
        -------
        state_vec : np.ndarray
            Vector representing initial state
        """
        state_string = self._get_state()
        self.state_string = state_string
        self.state_vec = self.to_vec(state_string)
        return self.state_vec

    @abstractmethod
    def _get_symbols(self):
        """
        Get equation symbols. e.g. symbols('x a b')

        Returns
        -------
        symbols
        """

    @abstractmethod
    def _get_terms(self):
        """
        Get terms for equation. e.g. [a, b, 0, 1]
        """

    @abstractmethod
    def _get_state(self):
        """
        Get environment state

        Example
        -------
        _, _, b = symbols('x a b')
        self.state_string = -b
        self.state_vec = self.to_vec(-b)
        return self.state_string

        Returns
        -------
        state_string : str
            State string representing environment state
        """

    @abstractmethod
    def _get_equation(self):
        """
        Simple linear equation

        Example
        -------
        x, a, b, = symbols('x a b')
        eqn = a * x + b
        return eqn

        Returns
        -------
        eqn : Object
            Equation object constructed from symbols
        """

    @abstractmethod
    def _get_feature_dict(self):
        """
        Get features at each node.

        Example
        -------
        keys = ['Add', 'Mul', 'Pow'] + ['x', 'a', 'b']
        return {key: -(i + 2) for i, key in enumerate(keys)}
        """

    @abstractmethod
    def find_loss(self, state):
        """
        Compute loss for the given state
        """

    def _make_physical_actions(self):
        """
        Operations x terms

        Returns
        -------
        actions : list
            List of operation, term pairs
        """

        illegal_actions = [[truediv, 0]]
        operations = [add, sub, mul, truediv, pow]
        terms = self._get_terms()
        actions = [[op, term] for op in operations for term in terms if
                   [op, term] not in illegal_actions]
        self.action_dim = len(actions)

        self.operations = operations
        self.terms = terms
        self.feature_dict = self._get_feature_dict()

        return actions

    def step(self, action: int):
        """
        Take step corresponding to the given action

        Parameters
        ----------
        action : int
            Action index corresponding to the entry in the action list
            constructed in _make_physical_actions

        Returns
        -------
        new_state_vec : np.ndarray
            New state vector after step
        reward : float
            Reward from taking this step
        done : bool
            Whether problem is solved or if maximum state dimension is reached
        info : dict
            Additional information
        """
        # action is 0,1,2,3, ...,  get the physical actions it indexes
        [operation, term] = self.actions[action]
        state_string = self.state_string
        new_state_string = operation(state_string, term)
        new_state_string = simplify(new_state_string)
        new_state_vec = self.to_vec(new_state_string)

        # Reward
        reward = self.find_reward(state_string, new_state_string)

        # Done
        done = False
        if self.too_long(new_state_vec):
            done = True

        # If loss is zero, you have solved the problem
        loss = self.find_loss(new_state_string)
        if loss == 0:
            done = True

        # Extra info
        info = {}

        # Update
        self.state_string = new_state_string

        logger.info('S, loss, reward, info = '
                    f'{self.state_string, loss, reward, info}')
        if loss == 0:
            logger.info(f'solution is: {self.state_string}')

        return new_state_vec, reward, done, info

    @classmethod
    def run(cls):
        """
        Run solver
        """
        env = cls()
        done = False
        action_dim = env.action_dim
        while not done:
            action = np.random.choice(action_dim)
            state, _, done, _ = env.step(action)
            loss = env.find_loss(env.state_string)
        if loss == 0:
            logger.info(f'solution is: {state}')
        else:
            logger.info('Terminating')

    def find_reward(self, state_old, state_new):
        """
        Reward is decrease in loss
        """
        loss_old = self.find_loss(state_old)
        loss_new = self.find_loss(state_new)
        return loss_old - loss_new

    def _walk(self, parent, expr, node_list, link_list):
        """
        Walk over the expression tree recursively creating nodes and links.

        Parameters
        ----------
        parent : Node
            Parent node
        expr : str
            State string
        node_list : list
            List of node dictionaries with 'id' and 'name' keys
        link_list : list
            List of link dictionaries with 'source' and 'target' keys
        """
        if expr.is_Atom:
            node = Node(str(expr), Id.get())
            node_list.append({"id": node.id, "name": node.name})
            link_list.append({"source": parent.id, "target": node.id})
        else:
            node = Node(str(type(expr).__name__), Id.get())
            node_list.append({"id": node.id, "name": node.name})
            link_list.append({"source": parent.id, "target": node.id})
            for arg in expr.args:
                self._walk(node, arg, node_list, link_list)

    def to_vec(self, expr):
        """
        Get state vector for given expression

        Parameters
        ----------
        expr : str
            State string representation

        Returns
        -------
        np.ndarray
            State vector array
        """
        node_list = []
        link_list = []

        self._walk(Node("Root", 0), expr, node_list, link_list)

        def pad_array(arr, length):
            """
            Pad array with zeros according the given length
            """
            if len(arr) < length:
                padded_arr = np.zeros(length)
                padded_arr[:len(arr)] = arr
                return padded_arr
            else:
                return arr

        # Create the graph from the lists of nodes and links:
        graph_json = {"nodes": node_list, "links": link_list}

        # Make node features. Map number to -1, else ...
        node_labels = {node['id']: node['name'] for node
                       in graph_json['nodes']}
        node_features = list(node_labels.values())
        node_features = np.array([int(self.feature_dict[key]) if key
                                 in self.feature_dict else int(key)
                                 for key in node_features])
        node_features = pad_array(node_features, int(0.25 * self.state_dim))

        for n in graph_json['nodes']:
            del n['name']

        # Make edge graph
        graph = json_graph.node_link_graph(graph_json, directed=True,
                                           multigraph=False)
        edge_vector = nx.to_numpy_array(graph).flatten()
        edge_vector = pad_array(edge_vector, int(0.75 * self.state_dim))
        state_vec = np.concatenate([node_features, edge_vector])

        return state_vec

    def to_graph(self, expr):
        """
        Make a graph plot of the internal representation of SymPy expression.
        """

        node_list = []
        link_list = []

        self._walk(Node("Root", 0), expr, node_list, link_list)

        # Create the graph from the lists of nodes and links:
        graph_json = {"nodes": node_list, "links": link_list}
        node_labels = {node['id']: node['name'] for node
                       in graph_json['nodes']}
        for n in graph_json['nodes']:
            del n['name']
        graph = json_graph.node_link_graph(graph_json, directed=True,
                                           multigraph=False)

        return graph, node_labels

    def plot_state_as_graph(self, expr):
        """
        Make a graph plot of the internal representation of SymPy expression.
        """
        graph, labels = self.to_graph(expr)
        pos = graphviz_layout(graph, prog="dot")
        nx.draw(graph.to_directed(), pos, labels=labels, node_shape="s",
                node_color="none", bbox={'facecolor': 'skyblue',
                                         'edgecolor': 'black',
                                         'boxstyle': 'round,pad=0.2'})

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

    # pylint: disable=unused-argument
    def render(self, mode='human'):
        """
        Print the state string representation
        """
        print(self.state)