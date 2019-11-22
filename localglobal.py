import pickle
import random
import numpy as np 
import pandas as pd
import seaborn as sns
import networkx as nx
from scipy.stats import norm
import matplotlib.pyplot as plt
from collections import Counter

### Helper methods

def create_subsets(size, scale = 1):
        """
        Creates a subset for later normal sampling based on provided scale.
        Can be used to update scale.

        Params:
            :int size:          The grid size.
            :int scale:         The scale (standard dev) of the random 
                                variable used to create subset vals. Mean = 0.
        
        Returns:
            None (changes subsets of Network)
        """

        # A copy paste job from Lapo's jupyter lab file.
        rv = norm(loc=0, scale=scale)
        x = np.linspace(-1, 1, size * 2 + 1)
        p = rv.pdf(x)

        peak = size
        cols = np.arange(size)

        subsets = []
        for col in cols:
            start = peak - col
            stop = start + size
            subset = np.array(p[start:stop])
            subset /= subset.sum()
            subsets.append(subset)
        
        for subset in subsets:
            assert np.isclose(subset.sum(), 1.0)
        
        return np.array(subsets)

def save_network(network):
    """
    Saves the network as a pickled object with naming scheme related to 
    network parameters.

    Parameters:
        :Network network:       The network object being saved.

    Returns:
        None.
    """

    size = str(network.L)
    scale = str(network.scale)

    filepath = "network_L" + size + "_S" + scale + ".pkl"

    fout = open(filepath, 'wb')
    pickle.dump(network, fout)
    fout.close()

def load_network(filepath):
    """
    Loads a network object from a given filepath.

    Parameters:
        :str filepath:      The filepath of the pickled object.

    Returns:
        :Network network:   The loaded network object.
    """

    fin = open(filepath, 'rb')
    network = pickle.load(fin)

    return network

def init_network_edges(N,av_k,sample_type):
    """
    Adds edges to the initialised empty network.

    Parameters:
        :N:                   Number of nodes in the network.
        :av_k:                Average degree.
        :sample_type:         The type of sampling method we want to use.

    Returns:
        None (adds edges to empty network).
    """

    for node in range(N):
        new_edges = np.random.normal(loc=av_k, scale=1.0)
        if sample_type == "local":
            network.local_sample(node, int(new_edges))
            
        
def add_network_edges(num_grow,N,sample_type):
    """
    Adds edges to the network.

    Parameters:
        :num_grow:            Number of edges to add in total.
        :N:                   Number of nodes in the network.
        :sample_type:         The type of sampling method we want to use.
            

    Returns:
        None (adds edges to network).
    """
    n_edge = 0
    while n_edge < num_grow:
        from_node = np.random.choice(N, num)
        num_edges = sum(map(len, network.adjlist.values()))
        if sample_type == "local":
            network.local_sample(from_node, 1)
        new_num_edges = sum(map(len, network.adjlist.values()))
        dif = new_num_edges - num_edges # because there are no paralell edges but we still want the tot num of edges fixed
        n_edge = n_edge + dif

def let_them_go(activity_state, firing_nodes): 
    """
    Adds activity to nodes that are neighbors of firing nodes.
    Firing nodes' activity goes down to 0.

    Parameters:
        :activity_state:          Array of activity state for each node.
        :firing_nodes:            Set of nodes that reached the treshold.

    Returns:
        New activity state and nodes that fired (we need for fitness).
    """
    node_fired = []
    for x in firing_nodes:
        nodes_that_receive = []
        nodes_that_receive.extend(network.adjlist.get(x))
        activity_state[nodes_that_receive] += 20
        activity_state[x] = 0
        node_fired.append(x)
    return activity_state, node_fired

def network_fitness(network, activity_state):
    """
    Calculates a fitness score for a network.

    Parameters:
        :network:          Network to calculate score for.
        :activity_state:   Array of activity state for each node.

    Returns:
        Fitness score.
    """
    node_fired_once = [] # list of nodes that fired
    activity_state[list(range(N))] = th # make all nodes fire
    firing_nodes = list(range(N))
    while len(firing_nodes) > 0:
        print("fired")
        new_activity_state, node_fired = let_them_go(activity_state, firing_nodes)
        node_fired_once.append(node_fired)
        activity_state = new_activity_state
        firing_nodes = np.where(activity_state >= th)[0]
    if len(node_fired_once) == 1:
        fitness = 0
    else: 
        fitness = len(Counter(node_fired_once[1]).keys())
    return (fitness)


### Main network object


class Network():
    """
    Basic network structure. Is represented by a fixed-distance euclidean grid.

    The network is comprised of two basic data types:
        1: A dictionary mapping each node ID to its position in the grid
        2: An adjacency list (also dictionary) showing which nodes are are
           connected to each other.

    Every other aspect of this object is devoted to somehow manipulating or
    displaying data from those two data structures.

    Attributes:
        tbd

    """

    def __init__(self, L = 10, scale = 1, ):
        """
        Initializes the Network by creating the relevant dictionary mapping
        and adjacency list. 

        Params: 
            :int L:         The initial side length of the network.
                            Grid size will not change after initialization!

            :int scale:     The intial scale of normally distributed sampling.
                            Subset scaling could change after initialization.
        Returns:

        """

        self.L = L
        self.scale = scale

        # Set empty adjlist, fill mapping.
        self.mapping = {}
        self.adjlist = {}
        for i in range(L):
            for j in range(L):

                # For each node ID, find coords and set empty adjlist.
                self.mapping[(self.L * i) + j] = (i,j)
                self.adjlist[(self.L * i) + j] = set()

        # Get reversed node mapping for ease.
        self.rapping = {v: k for k, v in self.mapping.items()}

        # Set subset probabilities.
        try:
            self.subsets = create_subsets(self.L, scale)
        except:
            print("\n\n Subsets must be LxL in size. \n\n")
            raise

    def local_sample(self, node, samples):
        """
        Chooses [samples] number of nodes to generate an edge to (from [node])
        given an existing set of subsets.

        Parameters:
            :int/tuple node:      The integer-ID of the node in question 
                                  or the coordinate of the node you want build from.

            :int samples:   The number of edges to build from that node.

        Returns:
            None. Makes edits to adj. list.
        """

        # Find coordinates (row, column) of this node:
        coords = self.mapping[node]

        # Find row, column of destination nodes
        to_row = np.random.choice(np.arange(self.L),
                                  size = samples,
                                  p = self.subsets[coords[0]])

        to_col = np.random.choice(np.arange(self.L),
                                  size = samples,
                                  p = self.subsets[coords[1]])

        # Update adjlist
        self.adjlist[node].update([to for to in self.L 
                              * to_row + to_col if node != to]) 

    def random_sample(self, node, samples):
        """
        Connects selected node with others at random.

        Parameters:
            :int/tuple node:        An int or coordinate representation of node
            :int samples:           Number of edges to create

        Returns:
            None. Makes changes to adj. list.
        """

        # Find points:
        points = np.hstack([np.arange(node), np.arange(node + 1, self.L ** 2)])

        # Find points without existing edges

        edges = np.random.choice(points, size=samples)
        
        self.adjlist[node].update([to for to in edges])

    def hybrid_sample(self, node, samples, p):
        """
        Samples with Lapo's hybrid method from the jupyter notebook.

        Mix of the previous two controlled by p -> fraction of local connections

        Parameters:
            :int node:          Node ID that we're sampling edges from.
            :int samples:       Number of nodes to sample to.
            :float p:           Fraction of local connections.

        Returns:
            None (Makes changes to adjlist)
        """

        assert 0 <= p and p <= 1
        _local = int(p * samples)
        _global = samples - _local

        # global
        global_points = np.hstack([np.arange(node), np.arange(node + 1, self.L ** 2)])
        global_edges = np.random.choice(global_points, size=_global)
        global_edges = [(node, to) for to in global_edges]

        # local
        col = node % self.L  # e.g. 11 % 5 -> col 1
        row = node // self.L  # e.g. 11 // 5 -> row 2
        local_points = np.arange(self.L)

        # Find points
        to_row = np.random.choice(local_points, p=self.subsets[row], size=_local)
        to_col = np.random.choice(local_points, p=self.subsets[col], size=_local)

        # reconstruct the node idx from rows and cols
        local_edges = [to for to in self.L * to_row + to_col if node != to]

        # Update adjlist.
        self.adjlist[node].update(local_edges)

    def draw_grid(self, size=10, labels=False):
        """
        Draws a grid from all adj list relationships currently saved.

        Parameters:
            :int size:          The size of the nodes.
            :bool labels:       Determines whether or not nodes are labelled.

        Returns:
            None (prints a figure).
        """
        
        # Build edges in networkx friendly format
        edges = []
        for i in self.adjlist.keys():
            if (len(self.adjlist[i]) > 0):
                for j in self.adjlist[i]:
                    edges.append((i,j))
                

        # Fix position mapping, because networkx is screwy.
        pos = {x:(self.mapping[x][1], self.L - self.mapping[x][0])
              for x in self.mapping.keys()}

        g = nx.DiGraph()
        g.add_nodes_from(self.mapping.keys())
        g.add_edges_from(edges)
        plt.figure(figsize=(16, 16))
        nx.draw(g, with_labels=labels, pos=pos, node_size=size)
        plt.show()
        

### Sample run

#Parameters
L = 20 #side of the grid
N = L*L #number of nodes
av_k = 5 #average degree in the beginning
init_edges = N*av_k # number of edges in the initial network
sample_type = "local"
locality = 1
th = 100 # minimum number of signals that initiate firing of node

#Init network and activity_state
network = Network(L, locality)
init_network_edges(init_edges,N,av_k,sample_type)
activity_state = np.zeros(N)

#Test network
fitness = network_fitness(network,activity_state)

#Test how fitness changes with locality
all_fitness = []
for i in np.arange(0.1, 10.0, 0.1):
    print(i)
    locality = i
    fitnesses = []
    for a in range(10):
        network = Network(L, locality)
        init_network_edges(N,av_k,sample_type)
        activity_state = np.zeros(N)
        fitness = network_fitness(network,activity_state)
        fitnesses.append(fitness)
    all_fitness.append(np.mean(fitnesses))

plt.plot(all_fitness)
        
"""
Sample call / use:

# Create network of L = 10, with normal dist. scale of 1

network = Network(L=10, scale=1)

# Check what the (row, col) coordinates of point 25 is, then sample 10 points
# to it.

network.mapping[25]
network.local_sample(25, 10)

# Display with numbers 

network.draw_grid(True, 10)
"""


