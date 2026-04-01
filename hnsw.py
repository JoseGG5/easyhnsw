from collections import defaultdict
from typing import Callable
import math
import random

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









def assign_layer(top_k):
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
    
    """ to match the hierarchical structure, hnsw will be a python dict 
    where each key will be fomratted as layer N and inside each layer there will
    a NSW graph"""
    
    # in the beginning, set max layer to -1
    L_max = -1
    
    hnsw = defaultdict(nx.Graph)

    for node_idx, x in enumerate(X):
                
        # get the layer assigned to the node
        layer = assign_layer(top_k=top_k)
        
        # if it's the first node we will append it from all layer between layer and 0
        if node_idx == 0:
            for layer_idx in range(layer, -1, -1):
                key = f"layer {layer_idx}"
                hnsw[key].add_node(node_idx, representation=x)
            
            # update max layer and jump to new data point
            L_max = layer
            continue
                
        
        # else, we perform the standard hnsw insert algorithm
        key = f"layer {L_max}"
        
        # get a random entrypoint from the max layer
        ep = select_entrypoint(nsw=hnsw[key])

        # from max layer to the one before the assigned searching for good entrypoints
        # note that if new layer is higher than L_max, this will not run (as expected)
        for layer_idx in range(L_max, layer, -1):
            
            key = f"layer {layer_idx}"
            
            found = greedy_search(
                nsw=hnsw[key],
                entry=ep,
                query_rpr=x,
                distance_metric=distance_metric
                )
            
            # create the set of possible nodes 
            # (just the node found with greedy and its neighbors)
            set_possible = [found]
            set_possible.extend(list(hnsw[key].neighbors(found)))
            
            # get the closest one from the possible list of candidates
            repr_candidates = [hnsw[layer_idx].nodes[node]["representation"] for node in set_possible]
            distances = distance_metric(x.reshape(1, -1), np.array(repr_candidates))[0]
            best_idx = np.argmin(distances)
            
            # update ep
            ep = set_possible[best_idx]
                
        
        # insert from the corresponding layer
        layer_to_insert_from = min(layer, L_max)
        for layer_idx in range(layer_to_insert_from, -1, -1):
            
            key = f"layer {layer_idx}"
            # same process than in a nsw graph
            # perform greedy search
            found = greedy_search(
                nsw=hnsw[key],
                entry=ep,
                query_rpr=x,
                distance_metric=distance_metric
                )

            # create the set of possible nodes (just the node found with greedy and its neighbors)
            set_possible = [found]
            set_possible.extend(list(hnsw[key].neighbors(found)))
            
            # get top-k with distance
            repr_topk = [hnsw[key].nodes[node]["representation"] for node in set_possible]
            distances = distance_metric(x.reshape(1, -1), np.array(repr_topk))[0]
            if len(distances) >= top_k: 
                topk_idxs = np.argpartition(distances, top_k - 1)[:top_k]
                nodes_connect = np.array(set_possible)[topk_idxs]
            else:
                nodes_connect = set_possible

            # create the node and add edges
            hnsw[key].add_node(node_idx, representation=x)
            for idx in nodes_connect:
                hnsw[key].add_edge(node_idx, idx)
            
            # next entry point for lower layer
            best_idx = np.argmin(distances)
            ep = set_possible[best_idx]
            
          
        """ update layer if needed and also, if layer is bigger than L_max
        we would have missed appending nodes from layer to last L_max, so we do that"""
        if layer > L_max:
            for layer_idx in range(L_max + 1, layer + 1):
                hnsw[layer_idx].add_node(node_idx, representation=x)
            L_max = layer
        
    return hnsw

    







if __name__ == "__main__":
    # simulate a batch of N representations with D dims each
    X = np.random.randn(N, D)
    
    # create a Hierarchical Navigable Small World data structure
    hnsw = create_hnsw(X, 8)
    
    

    
    