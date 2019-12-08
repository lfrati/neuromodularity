"""
This file provides classes and methods for networks. It is a cleaner,
refactored implementation of what currently exists in UniGaussianNets in the
backup branch.
"""

import time
import math
import json
import copy
import random

import numpy as np 
import pandas as pd 
import networkx as nx 
from abc import ABC, abstractmethod

from scipy.stats import norm
import matplotlib.pyplot as plt 

class Network(ABC):
    """
    Abstract class for networks of different varieties.
    """

    ### Abstract methods

    @abstractmethod
    def mutate(self, node):
        """
        Should make a single change to a single node / edge with respect
        to the type of network.
        """
        pass 

    @abstractmethod
    def initialize(self):
        """
        Should somehow initialize the network.
        """
        pass 

    @abstractmethod
    def evaluate(self):
        """
        Should determine the fitness of the network.
        """
        pass 

    ### Concrete methods

    def add_edges(self, node, num):
        """
        Adds edges to a node according to the sampling type.

        Parameters:
            :int node:          The node having edges added to it.
            :int num:           The number of edges added.

        Returns:
            None
        """

        # Find number of samples available.
        num_samples = min(num, self.N - 1 - len(self.adjL[node]))

        if (self.stype == "gaussian"):
            subset = compute_gaussian_weights(self.mweights, node, self.adjL)

            samples = np.random.choice(
                a = self.nodes,
                p = subset,
                replace = False,
                size = num)

            self.adjL[node].update(samples)

        elif (self.stype == "uniform"):

            available_nodes = list(set(self.nodes) - self.adjL[node])
            samples = np.random.choice(
                a = availableNodes, 
                replace = False, 
                size = num)

            self.adjL[node].update(samples)

    def fire(self):
        """
        Fires nodes in network with respect to their existing activity state.
        
        Parameters:
            None

        Returns:
            None
        """

        # For each item being fired
        firing = np.where(self.fireworthy == True)[0]
        for i in firing:

            # Find all neighbors and increment activity state by fireweight
            neigh = list(self.adjL[i])

            if (len(neigh) > 0):
                self.astates[neigh] += self.fireweight

    def firenode(self, node):
        """
        Fires a single node in the network.

        Parameters: 
            :int node:          The node being fired.

        Returns:
            None
        """

        self.astates[list(self.adjL[node])] += self.fireweight

    def spike(self):
        """
        Fires all nodes in the network without regard for their existing states.

        Parameters:
            None

        Returns:
            None
        """

        for i in self.adjL.keys():

            # Find all neighbors and increment activity state by fireweight
            neigh = list(self.adjL[i])

            if (len(neigh) > 0):
                self.astates[neigh] += self.fireweight

    def update_states(self):
        """
        Updates the states of each node.

        Parameters:
            None

        Return:
            None
        """

        to_fire = np.where(self.astates > self.threshold)
        self.astates = np.zeros(self.N ** 2)
        self.fireworthy = np.tile(False, self.N ** 2)
        self.fireworthy[to_fire] = True


    def show_grid(self, labels, size):
        """
        Visualizes whole network grid using networkx.

        Params:
            :int size:      The size of the nodes labels.
            :bool labels:   Whether the nodes are or are not labelled.

        Returns:
            None. (prints a figure)
        """

        # Reformat edges for networkx friendly format.
        edges = []
        for i in self.adjL.keys():
            if len(self.adjL[i]) > 0:
                for j in self.adjL[i]:
                    edges.append((i, j))

        # Create node mapping:
        mapping = {}
        for i in range(self.N):
            for j in range(self.N):

                # For each node ID, find coords and set empty adjlist.
                mapping[(self.N * i) + j] = (i, j)

        # Fix mapping to play nicer with networkx
        pos = {x: (mapping[x][1], self.N - mapping[x][0]) for x in mapping.keys()}

        # Display figure
        g = nx.DiGraph()
        g.add_nodes_from(mapping.keys())
        g.add_edges_from(edges)
        plt.figure(figsize=(16, 16))
        nx.draw(g, with_labels= labels, pos=pos, node_size=size)
        plt.show()


### MAIN NETWORKING CLASSES


class NoCommunity(Network):
    """
    Creates a network without any community structure.

    The key idea behind a non-community network is that it can grow;
    new edges can be added as opposed to existing edges being changed.
    """

    def __init__(self, ID, N, threshold, fireweight, stype, mweights, **kwargs):
        """
        Initializes network with no community firing structure

        Parameters:
            :int ID:            Unique ID of that network.
            :int N:             Side length of network.
            :int threshold:     Input required for a node to fire.
            :int fireweight:    Output of a single node
            :str stype: The sampling type used (gaussian, uniform)
            :list mweights:     Master weights matrix used for sampling.

        Returns:
            None (constructor)
        """

        # Network function params
        self.N = N
        self.threshold = threshold 
        self.fireweight = fireweight
        self.stype = stype
        self.mweights = mweights

        self.nodes = np.arange(self.N ** 2)
        self.adjL = {key: set() for key in np.arange(self.N ** 2)}
        self.astates = np.zeros(self.N ** 2)
        self.fireworthy = np.tile(False, self.N ** 2)

        # Evolutionary params
        self.ID = ID
        self.age = 0
        self.fitness = 0
        self.mutations = 0

    def comspike(self, com = None):
        """
        Fires all the nodes in a community ignoring their activity
        state. Essentially a community-specific spike.

        Parameters:
            :int com:       Index of the com to spike. Default of random.

        Returns:
            None
        """

        if (com == None):
            com = random.randrange(len(self.communities))

        for i in self.communities[com]:
            self.firenode(i)


    def mutate(self, node = None):
        """
        Mutation accomplished adding a single edge somewhere.

        Parameters:
            :int node:          The node being selected, Random by default.

        Returns:
            None (changes adjL)
        """
        if (node == None):
            node = random.randrange(self.N ** 2)

        self.add_edges(node, 1)

    def initialize(self, k, num):
        """
        Randomly seeds the network with num nodes each creating
        an average of av_k edges.

        Parameters:
            :int k:             Degree of node.
            :int num:           Number of nodes being initialized.
        """

        nodes = random.sample(self.nodes, num, replace=False)

        for i in nodes:
            self.add_edges(i,k)

    def evaluate(self):
        """
        To be determined.
        """

class GaussianCommunity(Network):
    """
    Creates a community network that is initialized with a broader, Gaussian 
    sampling method. The underlying network communities are still strict, but
    the actual sampling of nodes within does not lead to isolated, fully connected
    communities.
    """

    def __init__(self, ID, com_side, coms_per_side, threshold, fireweight, stype, mweights, **kwargs):
        """
        Initializes GaussianCommunity. Number of nodes is determined by the size of
        the communities and how many there are per side of the network.

        Parameters:
            :int com_side:          The length of one side of a community.
            :int coms_per_side:     How many communities there are per side of a network.
            :int threshold:         Input required for a single node to fire.
            :int fireweight:        Output of a fired single node.
            :str stype:             The sampling type (gaussian, uniform)
            :list mweights:         Master weight sampling matrix.

        Returns:
            None (constructor)
        """

         # Network function params
        self.N = com_side * coms_per_side
        self.com_side = com_side
        self.coms_per_side = coms_per_side
        self.threshold = threshold 
        self.fireweight = fireweight
        self.stype = stype
        self.mweights = mweights

        self.nodes = np.arange(self.N ** 2)
        self.adjL = {key: set() for key in np.arange(self.N ** 2)}
        self.astates = np.zeros(self.N ** 2)
        self.fireworthy = np.tile(False, self.N ** 2)
        self.communities = make_communities(com_side, coms_per_side)

        # Evolutionary params
        self.ID = ID
        self.age = 0
        self.fitness = 0
        self.mutations = 0

        if (mweights.shape[0] != (self.N * 2) + 1):
            print(mweights.shape)
            print(self.N)
            print("Your community size does not match your master weights!")
            raise

    def comspike(self, com = None):
        """
        Fires all the nodes in a community ignoring their activity
        state. Essentially a community-specific spike.

        Parameters:
            :int com:       Index of the com to spike. Default of random.

        Returns:
            None
        """

        if (com == None):
            com = random.randrange(len(self.communities))

        for i in self.communities[com]:
            self.firenode(i)


    def mutate(self, node = None):
        """
        Mutations add an edge, but only at the expense of an existing edge.

        Parameters:
            :int node:          The node that is losing an edge. Random by def.

        Returns:
            None
        """

        if (node == None):
            node = random.randrange(self.N ** 2)

        # Delete random edge from adjL for that node.
        self.adjL[node].discard(random.randrange(len(self.adjL[node])))

        # Choose a new random node and give it a sampled edge.
        new = random.randrange(self.N ** 2)
        self.add_edges(new, 1)


    def initialize(self):
        """
        Initialize the network with the number of edges it would take to 
        make fully connected communities (e.g. each node connects to 
        com_side ** 2 - 1 nodes). However, this connection is done using
        local Gaussian sampling.

        Parameters:
            None

        Returns:
            None
        """

        edges = (self.com_side ** 2) - 1

        for i in self.communities:
            for j in i:
                self.add_edges(j, edges)

    def evaluate(self):

        # Start with spiking all nodes in a community
        self.comspike()
        self.update_states()

        # Find initial firing nodes
        total_fires = np.where(self.fireworthy == True)

        iters = 50
        while(iters != 0 and True in self.fireworthy):
            print(iters)
            self.fire()
            self.update_states()

            total_fires = np.append(total_fires, np.where(self.fireworthy == True))

            iters -= 1

        # Calculate fitness based on proportion of total firing nodes & coms
        prop_fired = len(total_fires) / (self.N ** 2 * 50)

        my_set_f = set(list(total_fires))
        for c in self.communities:

            #number of communities where all nodes fired
            comm_fired = 0 
            for c in self.communities:
                my_set_comm = set(c)

                #communities and firing intersection
                comm_f = my_set_f.intersection(my_set_comm) 
                if len(comm_f) == len(my_set_comm):
                    comm_fired += 1

        self.fitness = np.average(prop_fired) + comm_fired

class StrictCommunity(Network):

    def __init__(self, ID, com_side, coms_per_side, threshold, fireweight, stype, mweights, **kwargs):
        """
        Initializes StrictCommunity. Number of nodes is determined by the size of
        the communities and how many there are per side of the network. 

        Similar to Gaussian community, but initialization is accomplished by connecting each
        node in a community to each other.

        Parameters:
            :int com_side:          The length of one side of a community.
            :int coms_per_side:     How many communities there are per side of a network.
            :int threshold:         Input required for a single node to fire.
            :int fireweight:        Output of a fired single node.
            :str stype:             The sampling type (gaussian, uniform)
            :list mweights:         Master weight sampling matrix.

        Returns:
            None (constructor)
        """

         # Network function params
        self.N = com_side * coms_per_side
        self.com_side = com_side
        self.coms_per_side = coms_per_side
        self.threshold = threshold 
        self.fireweight = fireweight
        self.stype = stype
        self.mweights = mweights

        self.nodes = np.arange(self.N ** 2)
        self.adjL = {key: set() for key in np.arange(self.N ** 2)}
        self.astates = np.zeros(self.N ** 2)
        self.fireworthy = np.tile(False, self.N ** 2)
        self.communities = make_communities(com_side, coms_per_side)

        # Evolutionary params
        self.ID = ID
        self.age = 0
        self.fitness = 0
        self.mutations = 0

        if (mweights.shape[0] != (self.N * 2) + 1):
            print(mweights.shape)
            print(self.N)
            print("Your community size does not match your master weights!")
            raise

    def comspike(self, com = None):
        """
        Fires all the nodes in a community ignoring their activity
        state. Essentially a community-specific spike.

        Parameters:
            :int com:       Index of the com to spike. Default of random.

        Returns:
            None
        """

        if (com == None):
            com = random.randrange(len(self.communities))

        for i in self.communities[com]:
            self.firenode(i)

    def mutate(self, node = None):
        """
        Mutations add an edge, but only at the expense of an existing edge.

        Parameters:
            :int node:          The node that is losing an edge. Random by def.

        Returns:
            None
        """

        if (node == None):
            node = random.randrange(self.N ** 2)

        # Delete random edge from adjL for that node.
        self.adjL[node].discard(random.randrange(len(self.adjL[node])))

        # Choose a new random node and give it a sampled edge.
        new = random.randrange(self.N ** 2)
        self.add_edges(new, 1)

    def initialize(self):
        """
        Initializes network by connecting all nodes within isolated 
        communities.

        Parameters:
            None

        Returns:
            None
        """

        for i in self.communities:

            # For each node within that community
            for j in range(len(i)):

                # Connect it to every other node within the community
                self.adjL[i[j]].update(np.delete(i, j))

    def evaluate(self):
        
        # Start with spiking all nodes in a community.
        self.comspike()
        self.update_states()

        # Find initial firing nodes
        total_fires = np.where(self.fireworthy == True)

        iters = 50
        while(iters != 0 and True in self.fireworthy):
            self.fire()
            self.update_states()

            total_fires = np.append(total_fires, np.where(self.fireworthy == True))

            iters -= 1

        # Calculate fitness based on proportion of total firing nodes & coms
        prop_fired = len(total_fires) / (self.N ** 2 * 50)

        my_set_f = set(list(total_fires))
        for c in self.communities:

            #number of communities where all nodes fired
            comm_fired = 0 
            for c in self.communities:
                my_set_comm = set(c)

                #communities and firing intersection
                comm_f = my_set_f.intersection(my_set_comm) 
                if len(comm_f) == len(my_set_comm):
                    comm_fired += 1

        self.fitness = np.average(prop_fired) + comm_fired

### HELPER FUNCTIONS ###

def make_communities(community_side, communities_per_side): 
    """
    Compute indexes for communities on a lattice
    e.g.

    A A A B B B
    A A A B B B
    A A A B B B
    C C C D D D
    C C C D D D
    C C C D D D

    community_side       = 3
    community_size       = 3*3 = 9
    communities_per_side = 2
    num_communities      = 4
    tot nodes            = 4*9

    returns:    [
                    [0,1,2,6,7,8,12,13,14] -> A
                    [3,4,5,9,10,11,,15,16,17] -> B
                    ...
                ]

    Paramteres:
        :int community_side:            The side len of each community
        :int communities_per_side:      The number of communites on each side

    Returns:
        List of lists of nodes for each community (see Example above)
    """
    community_size = community_side * community_side
    communities = []
    seed_node = 0
    for i in range(communities_per_side):
        for j in range(communities_per_side):
            community = []
            for k in range(community_side):
                for z in range(community_side):
                    _id = (
                        communities_per_side * community_size * i
                        + community_side * j
                        + z
                        + k * (communities_per_side * community_side)
                    )
                    # print(f"{_id} ", end="")
                    community.append(_id)
                # print("- ", end="")
            communities.append(community)
    return np.array(communities)

def normal_marginals(side_len, locality):
    rv = norm(loc=0, scale=locality)
    x = np.linspace(-1, 1, side_len * 2 + 1)
    p = rv.pdf(x)
    peak = side_len
    cols = list(range(side_len))
    marginals = []
    for col in cols:
        start = peak - col
        stop = start + side_len
        subset = np.array(p[start:stop])
        subset /= subset.sum()
        marginals.append(subset)
    for subset in marginals:
        assert np.isclose(subset.sum(), 1.0)
    return marginals

def gen_gauss_weights(marginals):
    weights = []
    for i, col in enumerate(marginals):
        for j, row in enumerate(marginals):
            X, Y = np.meshgrid(row, col)
            w = X * Y
            w[i][j] = 0
            w /= w.sum()
            weights.append(w.flatten())
    return np.array(weights)

def gen_unif_weights(N):
    w = np.ones((N ** 2, N ** 2))
    return w / N ** 2

def make_master_weights(N,locality):
    rv = norm(loc=0, scale=locality)

    # size is twice as big because we are going to use NxN subsets of it
    x = np.linspace(-1, 1, N * 2+1) 

    # make marginal gaussians
    p = rv.pdf(x) 

    # use numpy magic to make them of the right shapes before combining them
    X, Y = np.meshgrid(p, p) 

    # compute the 2D gaussian by multiplying the marginals together
    w = X * Y 
    w /= w.sum()
    return w

def node_to_idxs(node, N):
    return node // N, node % N

def get_subset(W, node,N):
    assert 0 <= node and node < N*N, f"Node index out of bounds: {node}"
    i,j = node_to_idxs(node,N) # convert node index to row,col coordinates
    subset = W[N-i:N-i+N,N-j:N-j+N] # extract a subset of the master_weights
    return np.copy(subset) # make a copy to make sure subsequent manipulations don't affect the master

def compute_gaussian_weights(W,node,adjL):
    tmp,N = W.shape
    tmp,N = tmp//2, N//2 # recover side-len from the weigths matrix, yeah, I did't want to have an extra parameter going around
    assert tmp == N, f"Weights have not the expected shape: Expected ({N},{N}), got ({tmp},{N})"
    gauss = get_subset(W,node,N) # get the appropiate subset in the manner we have shown above
    i,j = node_to_idxs(node,N) # zero the node coords to avoid self loops
    gauss[i][j] = 0
    for neigh in adjL[node]: # go through the neighs in the adjlist and zero them
        i,j = node_to_idxs(neigh,N)
        gauss[i][j] = 0
    gauss = gauss / gauss.sum() # normalize everything to make sure we have probabilities
    return gauss.flatten() # flatten them to use with np.random.choice