"""Collection of useful functions."""
import random
from collections import deque, namedtuple
from collections.abc import Iterable
from typing import Any, Union

import networkx as nx
import numpy as np
import torch
from networkx.readwrite import json_graph
from sympy.core import Basic, Expr
from torch_geometric.utils.convert import from_networkx

from rl_equation_solver.utilities.operators import fraction

random.seed(42)


Experience = namedtuple(
    "Experience", ("state", "next_state", "action", "reward", "done", "info")
)


class Batch:
    """Graph Embedding or state vector Batch."""

    def __init__(self) -> None:
        """Initialize the batch."""
        self.experience: Experience
        self.states: Union[torch.Tensor, list]
        self.next_states: Union[torch.Tensor, list]
        self.actions: torch.Tensor
        self.rewards: torch.Tensor
        self.dones: torch.Tensor


class GraphEmbedding:
    """Graph embedding class. This is for embedding node features in matrix of
    fixed sizes.
    """

    def __init__(
        self,
        graph: nx.Graph,
        n_observations: int,
        n_features: int,
        device: torch.device,
    ) -> None:
        """Initialize the GraphEmbedding object.

        Parameters
        ----------
        graph : nx.Graph
            networkx graph representation.
        n_observations : int
            Number of observations to use for embedding
        n_features : int
            Number of features to use for onehot encoding
        device : torch.device
            Device to use for pytorch
        """
        self.graph = graph
        G = from_networkx(self.graph)
        self._x = G.x.to(device)
        self.adj = G.edge_index.to(device)

        _x, _onehot_values = encode_onehot(np.array(self._x.cpu()))

        # embed in larger constant size matricies
        max_i = min(_x.shape[0], n_observations)
        max_j = min(_x.shape[1], n_features)
        x = np.zeros((n_observations, n_features))
        x[:max_i, :max_j] = _x[:max_i:, :max_j]

        self.x = torch.tensor(x, device=device, dtype=torch.float32)
        self.onehot_values = torch.tensor(
            pad_array(list(_onehot_values.keys()), n_features),
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)


class Memory:
    """Stores the Experience Replay buffer."""

    def __init__(self, capacity: int) -> None:
        """Initialize the memory buffer.

        Parameters
        ----------
        capacity : int
            Number of elements the memory can hold
        """
        self.capacity = capacity
        self.memory: deque = deque([], maxlen=self.capacity)

    def push(
        self,
        state: Union[torch.Tensor, GraphEmbedding],
        next_state: Union[torch.Tensor, GraphEmbedding],
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        info: dict,
    ) -> None:
        """Save the Experience into memory."""
        self.memory.append(
            Experience(state, next_state, action, reward, done, info)
        )

    def sample(self, batch_size: int) -> Iterable[Any]:
        """Select a random batch of Experience for training."""
        return random.sample(self.memory, batch_size)

    def get_all(self, device: torch.device) -> tuple[Any, ...]:
        """Get entire memory."""
        out: dict = {k: [] for k in self.memory[0]._fields}
        for i in range(len(self.memory)):
            for j, k in enumerate(self.memory[i]._fields):
                out[k].append(self.memory[i][j])
        for k, v in out.items():
            if "state" in k:
                if self.has_graph(v):
                    out[k] = v
                else:
                    out[k] = torch.stack(v)
            elif "info" not in k:
                out[k] = torch.tensor(v, device=device)
        return tuple(out.values())

    def clear(self) -> None:
        """Clear all memory."""
        self.memory = deque([], maxlen=self.capacity)

    def __len__(self) -> int:
        """Get length of memory."""
        return len(self.memory)

    def has_graph(self, v) -> bool:
        """Check if list has GraphEmbedding."""
        has_graph = isinstance(v, GraphEmbedding) or isinstance(
            v[0], GraphEmbedding
        )
        return has_graph

    def get_batch(self, batch_size: int, device: torch.device):
        """Get sampled batch from memory."""
        experiences = self.sample(batch_size)
        batch = Batch()
        batch.experience = Experience(*zip(*experiences))
        states = list(batch.experience.state)
        next_states = list(batch.experience.next_state)
        batch.dones = torch.tensor(
            [int(s) for s in batch.experience.done], device=device
        )
        batch.actions = torch.cat(batch.experience.action)
        batch.rewards = torch.cat(batch.experience.reward)
        graph_check = self.has_graph(states) or self.has_graph(next_states)
        if not graph_check:
            batch.next_states = torch.cat(next_states)
            batch.states = torch.cat(states)
        else:
            batch.next_states = next_states
            batch.states = states
        return batch


class Id:
    """A helper class for autoincrementing node numbers."""

    counter = -1

    @classmethod
    def get(cls) -> int:
        """Get the node number."""
        cls.counter += 1
        return cls.counter

    @classmethod
    def reset(cls) -> None:
        """Reset counter."""
        cls.counter = -1


class Node:
    """Represents a single operation or atomic argument."""

    def __init__(self, label: str, expr_id: int) -> None:
        """Initialize a new node."""
        self.id = expr_id
        self.name = label

    def __repr__(self) -> str:
        """Representation of a node."""
        return self.name


def graph_walk(
    parent: Node, expr: Union[Expr, Basic], node_list: list, link_list: list
) -> None:
    """Walk over the expression tree recursively creating nodes and links.

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
    if parent.name == "Root":
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


def pad_array(arr: Union[np.ndarray, list], length: int) -> np.ndarray:
    """Pad array with zeros according the given length."""
    max_i = min((length, len(arr)))
    padded_arr = np.zeros(length)
    padded_arr[:max_i] = arr[:max_i]
    if len(arr) < length:
        return padded_arr
    else:
        return np.array(arr)


def to_vec(
    expr: Expr, feature_dict: dict, state_dim: int = 4096
) -> np.ndarray:
    """Get state vector for given expression.

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
    state_vec = np.concatenate([node_features, edge_vector], dtype=np.float32)

    return state_vec


def get_json_graph(expr: Expr) -> nx.Graph:
    """Make a json graph for the SymPy expression. Don't add meta data yet.

    Parameters
    ----------
    expr : str
        State string representation

    Returns
    -------
    networkx.Graph
    """
    node_list: list = []
    link_list: list = []

    graph_walk(Node("Root", -1), expr, node_list, link_list)

    # Create the graph from the lists of nodes and links:
    graph_json = {"nodes": node_list, "links": link_list}
    node_labels = {node["id"]: node["name"] for node in graph_json["nodes"]}
    for n in graph_json["nodes"]:
        del n["name"]
    graph = json_graph.node_link_graph(
        graph_json, directed=True, multigraph=False
    )
    for node in graph.nodes:
        graph.nodes[node]["name"] = node_labels.get(node, "Root")
    graph.remove_node(-1)
    return graph


def to_graph(expr: Expr, feature_dict: dict) -> nx.Graph:
    """Make a graph plot of the internal representation of SymPy expression.

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
        graph.nodes[node]["x"] = node_features[i]
    return graph


def parse_node_features(
    node_features: Union[np.ndarray, list], feature_dict: dict
) -> list:
    """Parse node features. Includes string to fraction parsing."""
    parsed_features: list[Union[int, float]] = []
    for key in node_features:
        if key in feature_dict:
            parsed_features.append(int(feature_dict[key]))
        else:
            parsed_features.append(fraction(key))
    return parsed_features


def get_node_features(graph: nx.Graph, feature_dict: dict) -> np.ndarray:
    """Get node features from feature dictionary. e.g. we can map the
    operations and terms to integeters: {add: 0, sub: 1, .. }.
    """
    node_labels = list(get_node_labels(graph).values())
    return np.array(parse_node_features(node_labels, feature_dict))


def get_node_labels(graph: nx.Graph) -> dict[int, str]:
    """Get node labels from graph. Labels must be stored as node attributes as
    graph.nodes[index]['name']. Includes None for nodes with no name.

    Parameters
    ----------
    graph : networkx.graph
        Networkx graph object with node['name'] attributes
    """
    node_labels = {k: graph.nodes[k]["name"] for k in graph.nodes}
    return node_labels


def encode_onehot(labels: Union[list, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Onehot encoding."""
    classes = set(labels)
    classes_dict = {
        c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
    }
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32
    )
    return labels_onehot, classes_dict
