"""Environment for linear equation solver"""

import gym
from gym import spaces
import numpy as np
from sympy import simplify, expand, symbols
from operator import add, sub, mul, truediv, pow
import logging

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


class Env(gym.Env):
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
        self.node_labels = None
        self.operations = None
        self.actions = None
        self.terms = None
        self.feature_dict = {}
        self._state_vec = None
        self._state_graph = None

        self.max_loss = 50
        self.state_dim = 4096
        self.equation = self._get_equation()
        self.actions = self._make_physical_actions()
        self.state = self._init_state()

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
        self._state_vec = self.to_vec(self.state_string)
        return self._state_vec

    @state_vec.setter
    def state_vec(self, value):
        """Set state_vec value"""
        self._state_vec = value

    @property
    def state_graph(self):
        """Get current state graph"""
        self._state_graph = self.to_graph(self.state_string)
        return self._state_graph

    @state_graph.setter
    def state_graph(self, value):
        """Set state_graph value"""
        self._state_graph = value

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

    def find_loss(self, state):
        """
        Compute loss for the given state

        Parameters
        ----------
        state : str
            String representation of the current state

        Returns
        -------
        loss : int
            Number of edges plus number of nodes in graph representation /
            expression_tree of the current solution approximation
        """
        x, *_ = self._get_symbols()
        solution_approx = simplify(expand(self.equation.replace(x, state)))
        if solution_approx == 0:
            loss = 0
        else:
            state_graph = self.to_graph(solution_approx)
            loss = state_graph.number_of_nodes()
            loss += state_graph.number_of_edges()

        return loss

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
        self.actions = actions
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
        new_state_string = operation(self.state_string, term)
        new_state_string = simplify(new_state_string)
        new_state_vec = self.to_vec(new_state_string)

        # Reward
        reward = self.find_reward(self.state_string, new_state_string)

        # Done
        done = False
        if self.too_long(new_state_vec):
            done = True

        # If loss is zero, you have solved the problem
        loss = self.find_loss(new_state_string)
        if loss == 0:
            done = True

        # Update
        self.state_string = new_state_string

        # Extra info
        info = {'loss': loss, 'reward': reward, 'state': self.state_string}

        logger.info(f'info = {info}')

        if loss == 0:
            logger.info(f'solution is: {self.state_string}')

        return new_state_vec, done, info

    def find_reward(self, state_old, state_new):
        """
        Reward is decrease in loss

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
        loss_old = self.find_loss(state_old)
        loss_new = self.find_loss(state_new)
        return loss_old - loss_new

    def _get_feature_dict(self):
        """Return feature dict representing features at each node"""
        keys = [op.__name__.capitalize() for op in self.operations]
        keys += [str(sym) for sym in self._get_symbols()]
        return {key: -(i + 2) for i, key in enumerate(keys)}

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

        graph = self.to_graph(expr)
        node_labels = self.get_node_labels(graph)
        node_features = list(node_labels.values())
        node_features = np.array([int(self.feature_dict[key]) if key
                                  in self.feature_dict else int(float(key))
                                  for key in node_features])
        node_features = pad_array(node_features, int(0.25 * self.state_dim))
        edge_vector = nx.to_numpy_array(graph).flatten()
        edge_vector = pad_array(edge_vector, int(0.75 * self.state_dim))
        state_vec = np.concatenate([node_features, edge_vector])

        return state_vec

    def get_node_labels(self, graph):
        """Get node labels from graph. Must be stored as node attributes as
        graph.nodes[index]['name']

        Parameters
        ----------
        graph : networkx.graph
            Networkx graph object with node['name'] attributes
        """
        node_labels = {k: graph.nodes[k].get('name', None)
                       for k in graph.nodes
                       if graph.nodes[k]['name'] is not None}
        return node_labels

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
        for node in graph.nodes:
            graph.nodes[node]['name'] = node_labels.get(node, None)

        return graph

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
