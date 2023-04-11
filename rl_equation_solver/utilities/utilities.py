"""Collection of useful functions"""
import numpy as np
import scipy.sparse as sp
from collections import namedtuple
import torch
from torch_geometric.utils.convert import from_networkx
import networkx as nx
from networkx.readwrite import json_graph
from networkx.drawing.nx_pydot import graphviz_layout

from rl_equation_solver.utilities.operators import fraction


Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward'))


class Batch:
    """Graph Embedding or state vector Batch"""

    def __init__(self):
        """Initialize the batch"""
        self.experience = None
        self.non_final_mask = None
        self.non_final_next_states = None
        self.non_final_next_states = None
        self.state_batch = None
        self.action_batch = None
        self.reward_batch = None

    @classmethod
    def __call__(cls, states, device):
        """Batch states for given set of states and send to device. States
        can be either instances of GraphEmbedding or np.ndarray"""
        batch = cls()
        batch.experience = Experience(*zip(*states))
        batch.non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.experience.next_state)),
            device=device, dtype=torch.bool)
        batch.non_final_next_states = [s for s in batch.experience.next_state
                                       if s is not None]
        batch.state_batch = [s for s in batch.experience.state
                             if s is not None]
        batch.action_batch = torch.cat(batch.experience.action)
        batch.reward_batch = torch.cat(batch.experience.reward)
        return batch


class Id:
    """A helper class for autoincrementing node numbers."""
    counter = -1

    @classmethod
    def get(cls):
        """
        Get the node number
        """
        cls.counter += 1
        return cls.counter

    @classmethod
    def reset(cls):
        """Reset counter"""
        cls.counter = -1


class Node:
    """Represents a single operation or atomic argument."""

    def __init__(self, label, expr_id):
        self.id = expr_id
        self.name = label

    def __repr__(self):
        return self.name


class VectorEmbedding:
    """Vector embedding class for embedding feature vector in vector of
    fixed size"""
    def __init__(self, vector, n_observations, device):
        self.vector = pad_array(vector, n_observations)
        self.vector = torch.tensor(self.vector, device=device,
                                   dtype=torch.float32)


class GraphEmbedding:
    """Graph embedding class for embedding adjacency matrix and node features
    in matrices of fixed sizes"""
    def __init__(self, graph, n_observations, n_features, device):
        G = from_networkx(graph)
        self._x = G.x.to(device)
        self.adj = G.edge_index.to(device)

        self._x, self.onehot_values = encode_onehot(np.array(self._x.cpu()))
        self.onehot_values = np.array(list(self.onehot_values.keys()))
        self.onehot_values = pad_array(self.onehot_values, n_features)
        self.x = np.zeros((n_observations, n_features))

        # embed in larger constant size matricies
        max_i = min(self._x.shape[0], n_observations)
        max_j = min(self._x.shape[1], n_features)
        self.x[:max_i, :max_j] = self._x[:max_i:, :max_j]

        self.x = torch.tensor(self.x, device=device, dtype=torch.float32)
        self.onehot_values = torch.tensor(self.onehot_values, device=device,
                                          dtype=torch.float32).unsqueeze(0)


def graph_walk(parent, expr, node_list, link_list):
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
    if parent.name == 'Root':
        Id.reset()

    if expr.is_Atom:
        node = Node(str(expr), Id.get())
        node_list.append({"id": node.id, "name": node.name})
        link_list.append({"source": parent.id, "target": node.id})

    else:
        node = Node(str(type(expr).__name__), Id.get())
        node_list.append({"id": node.id, "name": node.name})
        link_list.append({"source": parent.id, "target": node.id})
        for arg in expr.args:
            graph_walk(node, arg, node_list, link_list)


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


def to_vec(expr, feature_dict, state_dim=4096):
    """
    Get state vector for given expression

    Parameters
    ----------
    expr : str
        State string representation
    feature_dict : dict
        Dictionary mapping feature names to values
    state_dim : int
        Max length of state vector

    Returns
    -------
    np.ndarray
        State vector array
    """
    graph = get_json_graph(expr)
    node_features = get_node_features(graph, feature_dict)
    node_features = pad_array(node_features, int(0.25 * state_dim))
    edge_vector = nx.to_numpy_array(graph).flatten()
    edge_vector = pad_array(edge_vector, int(0.75 * state_dim))
    state_vec = np.concatenate([node_features, edge_vector])

    return state_vec


def get_json_graph(expr):
    """
    Make a graph plot of the internal representation of SymPy expression. Don't
    add meta data yet.

    Parameters
    ----------
    expr : str
        State string representation

    Returns
    -------
    networkx.Graph
    """
    node_list = []
    link_list = []

    graph_walk(Node("Root", -1), expr, node_list, link_list)

    # Create the graph from the lists of nodes and links:
    graph_json = {"nodes": node_list, "links": link_list}
    node_labels = {node['id']: node['name'] for node
                   in graph_json['nodes']}
    for n in graph_json['nodes']:
        del n['name']
    graph = json_graph.node_link_graph(graph_json, directed=True,
                                       multigraph=False)
    for node in graph.nodes:
        graph.nodes[node]['name'] = node_labels.get(node, 'Root')
    graph.remove_node(-1)
    return graph


def to_graph(expr, feature_dict):
    """
    Make a graph plot of the internal representation of SymPy expression.

    Parameters
    ----------
    expr : str
        State string representation
    feature_dict : dict
        Dictionary mapping feature names to values

    Returns
    -------
    networkx.Graph
    """
    graph = get_json_graph(expr)
    node_features = get_node_features(graph, feature_dict)
    for i, node in enumerate(list(graph.nodes)):
        graph.nodes[node]['x'] = node_features[i]
    return graph


def parse_node_features(node_features, feature_dict):
    """Parse node features. Includes string to fraction parsing"""
    parsed_features = []
    for key in node_features:
        if key in feature_dict:
            parsed_features.append(int(feature_dict[key]))
        else:
            parsed_features.append(fraction(key))
    return parsed_features


def get_node_features(graph, feature_dict):
    """Get node features from feature dictionary. e.g. we can map the
    operations and terms to integeters: {add: 0, sub: 1, .. }"""
    node_labels = get_node_labels(graph)
    node_features = list(node_labels.values())
    node_features = np.array(parse_node_features(node_features, feature_dict))
    return node_features


def get_node_labels(graph):
    """Get node labels from graph. Must be stored as node attributes as
    graph.nodes[index]['name']. Includes None for nodes with no name

    Parameters
    ----------
    graph : networkx.graph
        Networkx graph object with node['name'] attributes
    """
    node_labels = {k: graph.nodes[k]['name'] for k in graph.nodes}
    return node_labels


def plot_state_as_graph(expr):
    """
    Make a graph plot of the internal representation of SymPy expression.
    """
    graph = to_graph(expr)
    labels = get_node_labels(graph)
    pos = graphviz_layout(graph, prog="dot")
    nx.draw(graph.to_directed(), pos, labels=labels,
            node_shape="s", node_color="none",
            bbox={'facecolor': 'skyblue', 'edgecolor': 'black',
                  'boxstyle': 'round,pad=0.2'})


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_adjacency_matrix_custom(graph):
    """Build adjacency matrix from graph edges and labels"""
    edges = np.array(graph.edges)
    labels = np.array(graph.nodes)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def build_adjacency_matrix(graph):
    """Build adjacency matrix from graph edges and labels"""
    return nx.adjacency_matrix(graph)


def encode_onehot(labels):
    """Onehot encoding"""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot, classes_dict
