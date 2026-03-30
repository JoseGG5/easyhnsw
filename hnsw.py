
from typing import Callable
import random
import math

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_distances
from matplotlib import pyplot as plt

""" Simplified version of a HNSW-like data structure. The core idea is the same, but note
    that we are performing random entrypoint selection, and we don't have a tracking
    of the degree of each node. However, this version is more than enough to grasp
    what a HNSW is how it is created. """

N = 100
D = 128

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




def create_nsw(X: np.ndarray, top_k: int, distance_metric: Callable = cosine_distances) -> nx.Graph:    
    nsw = nx.Graph()

    for i, x in enumerate(X):
        # if graph is empty, append just the first node
        if nsw.number_of_nodes() == 0:
            nsw.add_node(i, representation=x)
        
        else:
            # select an entrypoint (for simplicity just a random one)
            ep = select_entrypoint(nsw=nsw)

            # perform greedy search
            found = greedy_search(nsw=nsw, entry=ep, query_rpr=x, distance_metric=distance_metric)

            # create the set of possible nodes (just the node found with greedy and its neighbors)
            set_possible = [found]
            set_possible.extend(list(nsw.neighbors(found)))
            
            # get top-k with distance
            repr_topk = [nsw.nodes[node]["representation"] for node in set_possible]
            distances = distance_metric(x.reshape(1, -1), np.array(repr_topk))[0]
            if len(distances) >= top_k + 1: 
                topk_idxs = np.argpartition(distances, top_k - 1)[:top_k]
                nodes_connect = np.array(set_possible)[topk_idxs]
            else:
                nodes_connect = set_possible

            # create the node and add edges
            nsw.add_node(i, representation=x)
            for idx in nodes_connect:
                nsw.add_edge(i, idx)

    return nsw




def assign_layer(top_k):
    """ Assign each node to a layer following a given distribution.
    The idea is that low layers will contain most or
    all nodes and upper layers will have a small number of nodes.
    This introduces the HIERARCHICAL concept."""

    """ Note that ideally, our NSW would implement some mechanism
    to control the number of neighbors per node and that shoud be the parameter that is passed
    in this function. However, as our nsw implementation only handles the number of neighbors
    connected to each node when inserting it (and not the maximum numbers of neighbors during the whole
    index creation), we use this as the parameter to assign layer."""
    
    mL = 1 / math.log(top_k)
    return int(-math.log(random()) * mL)




if __name__ == "__main__":
    # simulate a batch of N representations with D dims each
    X = np.random.randn(N, D)

    # create a Hierarchical Navigable Small World data structure
    nsw = create_nsw(X=X, top_k=3)

    # plot it
    nx.draw(nsw, with_labels=True)
    plt.show()

    