from collections import defaultdict
from typing import Callable
import math
import random

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_distances
from matplotlib import pyplot as plt

""" Simplified version of a HNSW-like data structure.

    The core idea is preserved, but note that:
    
    1. We use random entrypoint selection instead of maintaining a global entry point,
       which differs from the original HNSW algorithm.
    
    2. We do not enforce a maximum number of connections per node. The parameter `top_k`
       only controls how many neighbors a node connects to at insertion time, but existing
       nodes are not pruned.
    
    3. We do not use efConstruction. In the original HNSW algorithm, insertion involves
       exploring a larger neighborhood and keeping track of the efConstruction best candidates.
       Here, for simplicity, we only perform a greedy search, select the closest node found,
       and connect to it and its neighbors. This means that as our search is worse, long connections
       have a smaller chance to appear.
    
    Despite these simplifications, this implementation captures the main intuition behind
    HNSW and is sufficient to understand how the structure is built.
"""


def select_entrypoint(nsw: nx.Graph) -> int:
    """Naive random selection

    Args:
        nsw (nx.Graph): The nsw graph

    Returns:
        int: The chosen node
    """
    all_nodes = list(nsw.nodes())
    return random.choice(all_nodes)


def greedy_search(nsw: nx.Graph, entry: int, query_rpr: np.ndarray, distance_metric: Callable) -> int:
    """Performs greedy search to find the closest neighbor.

    The idea is that we compute the distance from all the neighbors of our current node
    and check if we would get close to the target representation by moving to one of those nodes.
    
    If we don't get close, we simply return our current node.

    Args:
        nsw (nx.Graph): The nsw graph
        entry (int): The index of the entry node
        query_rpr (np.ndarray): The target (query) representation
        distance_metric (Callable): The function to use

    Returns:
        int: The index of the node found
    """

    to_visit = list(nsw.neighbors(entry))

    # at the beginning we compute the distance between our entrypoint and the target representation
    entry_repr = nsw.nodes[entry]["representation"]
    best_total = distance_metric(entry_repr.reshape(1, -1), query_rpr.reshape(1, -1))[0, 0]

    current = entry

    while to_visit:
        
        distances = []
        for idx_node in to_visit:  # iter to neighbors of current node
            data_node = nsw.nodes[idx_node]
            representation = data_node["representation"]
            
            distance = distance_metric(representation.reshape(1, -1), query_rpr.reshape(1, -1))[0, 0]
            distances.append(distance)

        
        if not distances:
            return current
        
        # select the closest node
        distances = np.array(distances)
        min_idx = np.argmin(a=distances)
        min_value = np.min(distances)

        # we can't get closer, just return our current node
        if min_value >= best_total:
            return current
        
        # move to a new node and continue searching
        else:
            current = to_visit[min_idx]
            to_visit = list(nsw.neighbors(current))
            best_total = min_value
    
    return current


def assign_layer(top_k: int):
    """ Assign each node to a layer following a given distribution.
    The idea is that low layers will contain most or
    all nodes and upper layers will have a small number of nodes.
    This introduces the HIERARCHICAL concept."""

    """ Note that ideally, our NSW would implement some mechanism
    to control the number of neighbors per node and that shoud be the parameter that is passed
    in this function. However, as our nsw implementation only handles the number of neighbors
    connected to each node when inserting it (and not the maximum numbers of neighbors during 
    the whole index creation), we use this as the parameter to assign layer."""
    
    mL = 1 / math.log(top_k)
    return int(-math.log(np.random.rand()) * mL)


def create_hnsw(X: np.ndarray, top_k: int, distance_metric: Callable = cosine_distances) -> dict:
    """
    Function used to build the hnsw data structure.

    Parameters
    ----------
    X : np.ndarray
        The dataset from which we will build our data structure.
    top_k : int
        The parameter that controls to how many nodes each node gets connected at insert time.
    distance_metric : Callable, optional
        Function used to compute the distance metric. The default is cosine_distances.

    Returns
    -------
    dict
        A dict where each key is a layer and the value associated is the NSW data structure.

    """
    hnsw = defaultdict(nx.Graph)
    L_max = -1  # in the beginning we don't have any layers

    for node_idx, x in enumerate(X):
        
        # get the layer assigned to the node
        layer = assign_layer(top_k=top_k)

        # if it's the first node we will append it from all layer between layer and 0
        if node_idx == 0:
            for layer_idx in range(layer, -1, -1):
                key = f"layer {layer_idx}"
                hnsw[key].add_node(node_idx, representation=x)
            L_max = layer
            continue

        # we begin with the typical hnsw algorithm
        # we start from max layer and descent to layer - 1 looking for the
        # best entrypoints
        key = f"layer {L_max}"
        ep = select_entrypoint(hnsw[key])

        # greedy descent from L_max to layer -1
        # note that if L_max < layer, this is not executed (as expected) as we
        # don't need greedy descent and would only need to insert from layer to 0
        for layer_idx in range(L_max, layer, -1):
            key = f"layer {layer_idx}"

            found = greedy_search(
                nsw=hnsw[key],
                entry=ep,
                query_rpr=x,
                distance_metric=distance_metric
            )

            # refine entrypoint by looking at the closest one from the possible set
            set_possible = [found] + list(hnsw[key].neighbors(found))
            repr_candidates = [hnsw[key].nodes[n]["representation"] for n in set_possible]
            distances = distance_metric(x.reshape(1, -1), np.array(repr_candidates))[0]

            ep = set_possible[np.argmin(distances)]

        # now we proceed to insert from layer - 1 to 0 with the best ep found
        for layer_idx in range(min(layer, L_max), -1, -1):
            key = f"layer {layer_idx}"

            found = greedy_search(
                nsw=hnsw[key],
                entry=ep,
                query_rpr=x,
                distance_metric=distance_metric
            )

            set_possible = [found] + list(hnsw[key].neighbors(found))

            repr_candidates = [hnsw[key].nodes[n]["representation"] for n in set_possible]
            distances = distance_metric(x.reshape(1, -1), np.array(repr_candidates))[0]

            if len(distances) > top_k:
                topk_idxs = np.argpartition(distances, top_k - 1)[:top_k]
                nodes_connect = list(np.array(set_possible)[topk_idxs])
            else:
                nodes_connect = set_possible

            # add node and connect to the top k closest
            hnsw[key].add_node(node_idx, representation=x)
            for n in nodes_connect:
                hnsw[key].add_edge(node_idx, n)

            # update entrypoint
            ep = set_possible[np.argmin(distances)]

        # now, if layer is bigger than L_max, then we would have only inserte
        # from L_max to 0, but from layer to L_max we have not inserted so we
        # proceed to do that and update L_max
        if layer > L_max:
            for layer_idx in range(L_max + 1, layer + 1):
                key = f"layer {layer_idx}"
                hnsw[key].add_node(node_idx, representation=x)
            L_max = layer

    return hnsw



if __name__ == "__main__":
    # simulate a batch of N representations with D dims each
    N = 1000
    D = 128
    X = np.random.randn(N, D)
    
    # create a Hierarchical Navigable Small World data structure
    hnsw = create_hnsw(X, 8)
    
    for layer, graph in hnsw.items():
        # plot it
        fig, ax = plt.subplots()
        nx.draw(graph, ax=ax, with_labels=True)
        ax.set_title(layer)
        plt.show()
            
    assert hnsw["layer 0"].number_of_nodes() == N, "LAYER 0 IS UNCOMPLETED"
    
    