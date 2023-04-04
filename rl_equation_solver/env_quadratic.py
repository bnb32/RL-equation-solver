import gym

import numpy as np
import networkx as nx

from gym import spaces
from networkx.readwrite import json_graph
from sympy import symbols, simplify, expand
from operator import add, sub, mul, truediv, pow
from networkx.drawing.nx_pydot import graphviz_layout


class Env(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self):

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
        state_string = self._get_state()
        self.state_string = state_string
        self.state_vec = self.to_vec(state_string)
        return state_string
        
    def _get_state(self):
        x, a, b, c = symbols('x a b c')
        self.state_string = -b
        self.state_vec = self.to_vec(-b)
        return self.state_string 

    def _get_equation(self):
        """ Simple quadratic """
        x, a, b, c = symbols('x a b c')
        eqn = a*x + b
        return eqn

    def _make_physical_actions(self):
        """ Operations x terms """
        
        illegal_actions = [[truediv,0]]
        
        x, a, b, c = symbols('x a b c')
        operations = [add, sub, mul, truediv, pow]
        terms = [a,b,c,0,1]
        actions = [[op,term] for op in operations for term in terms if [op,term] not in illegal_actions]
        self.action_dim = len(actions)
        
        self.operations = operations
        self.terms = terms

        # This is for the features at each node
        keys = ['Add','Mul','Pow'] + ['x','a','b','c']
        self.feature_dict = {key:-(i+2) for i,key in enumerate(keys)}
                
        return actions

    def step(self, action: int):

        # action is an index, get the physical actions it indexes
        [operation,term] = self.actions[action]
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
        if self.find_loss(new_state_string) == 0:
            done = True

        # Extra info
        info = {}

        # Update
        self.state_string = new_state_string
        
        return new_state_vec, reward, done, info

    def find_reward(self,state_old, state_new):
        """ Reward is decrease in loss """

        L_old = self.find_loss(state_old)
        L_new = self.find_loss(state_new)
        reward = L_old - L_new
        return reward

    def find_loss(self, state):
        """ Stuff here """

        x, a, b, c = symbols('x a b c') 
        solution_approx = simplify(expand(self.equation.replace(x,state)))
        if solution_approx == 0:
            loss = 0
        else:
            state_graph = self.to_graph(solution_approx)
            loss = state_graph.number_of_nodes() + state_graph.number_of_edges()
            
        return loss
    

    def to_vec(self, expr):

        node_list = []
        link_list = []

        class Id:
            """A helper class for autoincrementing node numbers."""
            counter = 0

            @classmethod
            def get(cls):
                cls.counter += 1
                return cls.counter

        class Node:
            """Represents a single operation or atomic argument."""

            def __init__(self, label, expr_id):
                self.id = expr_id
                self.name = label

            def __repr__(self):
                return self.name
            
            
        def _walk(parent, expr):
            """Walk over the expression tree recursively creating nodes and links."""
            if expr.is_Atom:
                node = Node(str(expr), Id.get())
                node_list.append({"id": node.id, "name": node.name})
                link_list.append({"source": parent.id, "target": node.id})
            else:
                node = Node(str(type(expr).__name__), Id.get())
                node_list.append({"id": node.id, "name": node.name})
                link_list.append({"source": parent.id, "target": node.id})
                for arg in expr.args:
                    _walk(node, arg)

        _walk(Node("Root", 0), expr)

        def pad_array(arr, length):
            if len(arr) < length:
                padded_arr = np.zeros(length)
                padded_arr[:len(arr)] = arr
                return padded_arr
            else:
                return arr

        # Create the graph from the lists of nodes and links:    
        graph_json = {"nodes": node_list, "links": link_list}

        # Make node features. Map number to -1, else ...
        node_labels = {node['id']: node['name'] for node in graph_json['nodes']}
        node_features = list(node_labels.values())
        node_features = np.array([int(self.feature_dict[key]) if key in self.feature_dict else int(key) for key in node_features])
        node_features = pad_array(node_features, int(0.25*self.state_dim))

        for n in graph_json['nodes']:
            del n['name']

        # Make edge grahp
        graph = json_graph.node_link_graph(graph_json, directed=True, multigraph=False)
        edge_vector = nx.to_numpy_array(graph).flatten()
        edge_vector =  pad_array(edge_vector, int(0.75*self.state_dim))
        state_vec = np.concatenate([node_features, edge_vector])

        return state_vec


    def to_graph(self, expr):
        """
        Make a graph plot of the internal representation of SymPy expression.
        """

        node_list = []
        link_list = []

        class Id:
            """A helper class for autoincrementing node numbers."""
            counter = 0

            @classmethod
            def get(cls):
                cls.counter += 1
                return cls.counter

        class Node:
            """Represents a single operation or atomic argument."""

            def __init__(self, label, expr_id):
                self.id = expr_id
                self.name = label

            def __repr__(self):
                return self.name
            
            
        def _walk(parent, expr):
            """Walk over the expression tree recursively creating nodes and links."""
            if expr.is_Atom:
                node = Node(str(expr), Id.get())
                node_list.append({"id": node.id, "name": node.name})
                link_list.append({"source": parent.id, "target": node.id})
            else:
                node = Node(str(type(expr).__name__), Id.get())
                node_list.append({"id": node.id, "name": node.name})
                link_list.append({"source": parent.id, "target": node.id})
                for arg in expr.args:
                    _walk(node, arg)

        _walk(Node("Root", 0), expr)

        # Create the graph from the lists of nodes and links:    
        graph_json = {"nodes": node_list, "links": link_list}
        node_labels = {node['id']: node['name'] for node in graph_json['nodes']}
        for n in graph_json['nodes']:
            del n['name']
        graph = json_graph.node_link_graph(graph_json, directed=True, multigraph=False)

        return graph
    
    def plot_state_as_graph(self, expr):
        """
        Make a graph plot of the internal representation of SymPy expression.
        """

        node_list = []
        link_list = []

        class Id:
            """A helper class for autoincrementing node numbers."""
            counter = 0

            @classmethod
            def get(cls):
                cls.counter += 1
                return cls.counter

        class Node:
            """Represents a single operation or atomic argument."""

            def __init__(self, label, expr_id):
                self.id = expr_id
                self.name = label

            def __repr__(self):
                return self.name
            
            
        def _walk(parent, expr):
            """Walk over the expression tree recursively creating nodes and links."""
            if expr.is_Atom:
                node = Node(str(expr), Id.get())
                node_list.append({"id": node.id, "name": node.name})
                link_list.append({"source": parent.id, "target": node.id})
            else:
                node = Node(str(type(expr).__name__), Id.get())
                node_list.append({"id": node.id, "name": node.name})
                link_list.append({"source": parent.id, "target": node.id})
                for arg in expr.args:
                    _walk(node, arg)

        _walk(Node("Root", 0), expr)

        # Create the graph from the lists of nodes and links:    
        graph_json = {"nodes": node_list, "links": link_list}
        node_labels = {node['id']: node['name'] for node in graph_json['nodes']}
        for n in graph_json['nodes']:
            del n['name']
        graph = json_graph.node_link_graph(graph_json, directed=True, multigraph=False)

        pos = graphviz_layout(graph, prog="dot")
        nx.draw(graph.to_directed(), pos, labels=node_labels, node_shape="s",  
                node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))

        return 

    def too_long(self, state):
        return len(state) > self.state_dim    

    def render(self, mode='human'):
        print(self.state)
        

# env = Env()
# done = False
# action_dim = env.action_dim
# actions = list(range(action_dim))
# while not done:
#     action = np.random.choice(action_dim)
#     state, reward, done, info = env.step(action)
#     loss = env.find_loss(env.state_string)
#     print(f'S, loss = {env.state_string,loss}')
# if loss == 0:
#     print(f'solution is: {state}')
# else:
#     print(f'Terminating')