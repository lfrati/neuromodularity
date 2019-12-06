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

import matplotlib.pyplot as plt 

from scipy.stats import norm

class Network:
    """
    The core network class.

    Attributes:

    """

    def __init__(self, **kwargs):
        """
        Constructor for network object.

        Parameters:
            :int N:             Side length of adjM of the network.
            :int threshold:     The threshold of firing nodes within the network
            :int fireweight:    Strength of a node firing.
            :str stype:         Sampling type (gaussian or uniform)

            :float locality:    Locality used in guassian / local sampling.
            :tuple comshape:    Community shape in set-community networks.

        Returns:
            None (constructor)
        """

        # Set some defaults
        kwargs.setdefault("N", 10)
        kwargs.setdefault("threshold", 100)
        kwargs.setdefault("fireweight", 20)
        kwargs.setdefault("stype", "gaussian")
        kwargs.setdefault("locality", 0.25)
        kwargs.setdefault("comshape", None)

        self.__dict__.update(kwargs)

        kwargs.setdefault("mweights", make_master_weights(self.N, self.locality))

        self.__dict__.update(kwargs)

        # Determine whether community structure exists:
        if (self.comshape  != None):
            self.communities = make_communities(self.comshape[0], 
                self.comshape[1])
            
            if (self.N != self.comshape[0] * self.comshape[1]):
                print("Overriding expected N value.")
                self.N = self.comshape[0] * self.comshape[1]

        # Determine core network attributes
        self.adjL = {key: set() for key in np.arange(self.N ** 2)}
        self.nodes = np.arange(self.N ** 2)

        self.astates = np.zeros(self.N ** 2)
        self.fireworthy = np.tile(False, self.N ** 2)

        self.age = 0
        self.muts = 0
        self.fitness = 0

        # Determine weights for sampling methods
        if (self.stype == "gaussian"):
            marginals = normal_marginals(self.N, self.locality)
            self.weights = gen_gauss_weights(marginals)
            self.original_weights = np.copy(self.weights)

        elif (self.stype == "uniform"):
            self.weights = gen_unif_weights(self.N)
            self.original_weights = np.copy(self.weights)
            

    def add_edges(self, node, num):
        """
        Add edges certain number of edges to a network according to that
        network's selection scheme (stype).

        Parameters:
            :int node:          The node having edges added.
            :int num:           The number of edges being added to that node.

        Returns:
            None (modifies adjL)
        """

        num_samples = min(num, self.N - 1 - len(self.adjL[node]))

        if (self.stype == "gaussian"):

            subset = compute_gaussian_weights(self.mweights, node, self.adjL)

            samples = np.random.choice(
                a = self.nodes,
                p = subset,
                replace = False,
                size = num
            )

            self.adjL[node].update(samples)

        elif (self.stype == "uniform"):
            availableNodes = list(set(self.nodes) - self.adjL[node])
            samples = np.random.choice(
                a = availableNodes, 
                replace = False, 
                size = num)

            self.adjL[node].update(samples)

    def mutate(self):
        """
        A simple wrapper for add_edges: It chooses a random node and 
        performs a sample on it. Used in evolution.

        Parameters:
            None

        Returns:
            None
        """
        
        if (self.comshape == None):
            # Add a single edge to a random node, increment age
            node = random.randrange(0, self.N ** 2)
            self.add_edges(node, 1) 
            self.muts += 1
        else:
            #Mutation happens by both adding and removing an edge of a chosen node (i.e. rewiring)

            # Add a single edge to a random node, increment age
            node = random.randrange(0, self.N ** 2)
            out_node_old = random.sample(self.adjL[node],1)


            # sample new destination before undoing the old one to avoid re-picking it
            self.add_edges(node, 1)


            #self.weights[node][sample] = 0 put back original weight
            self.weights[node][out_node_old] = self.original_weights[node][out_node_old]

            # self.adjL[node].add(sample)
            self.adjL[node].discard(out_node_old[0])
              
            self.muts += 1
    
    def fire(self, iters):
        """
        Fires every node in the graph.

        Parameters:
            None

        Returns:
            None
        """

        # For each item being fired
        firing = np.where(self.fireworthy == True)[0]
        for i in firing:

            # Find all neighbors and increment activity state += 20
            neigh = list(self.adjL[i])

            if len(neigh) > 0:
                if iters == 50: #the first time the network fires it is a bigger signal
                    self.astates[neigh] += self.threshold
                else:
                    self.astates[neigh] += 20

    def initialize(self, nprop = .1, avk = 1):
        """
        Initializes a subset of the network with nodes of an average degree.

        Paramaters:
            :float nprop:       Proportion of network starting with edges.
            :int avk:           Average degree of nodes starting with edges.

        Return:
            None
        """

        # If non-community based, randomly initialize
        if (self.comshape == None):
            nodes = np.random.choice(self.nodes, 
                size = int(len(self.nodes) * nprop))

            for node in nodes:
                nedges = int(np.random.lognormal(avk))
                self.add_edges(node, nedges)
                
        else: #create random network but with communities
            
            tot_num_edges = (len(self.communities[0]) * (len(self.communities[0])-1)) * len(self.communities)
            
            for node in self.nodes:
                for _ in range(int(tot_num_edges/len(self.nodes))):
                    self.add_edges(node, 1)

        '''        
        # If community based, connect all communities
        else: 
            # For each community
            for i in self.communities:

                # For each node within that community
                for j in range(len(i)):

                    # Connect it to every other node within the community
                    self.adjL[i[j]].update(np.delete(i, j))
        '''

    def evaluate(self):
        
        if (self.comshape != None):
        
            # Set all states as not firing
            self.fireworthy = np.array([False] * (self.N ** 2))
            
            # Set certain community as firing
            #self.fireworthy[self.communities[np.random.randint(len(self.communities))]] = True
            self.fireworthy[self.communities[0]] = True

            props = []

            iters = 50 # measuring fitness after 50 iterations
            while iters > 0: #and len(np.where(self.fireworthy == True)[0]) > 0:
                # Update activity states
                self.fire(iters)
                #print("fired")

                # Save all nodes that will fire
                firing = np.where(self.astates >= self.threshold)[0]

                # Find proportion of nodes that fire
                prop = len(firing) / self.N ** 2
                props.append(prop)

                # Reset firing states, keep those that are > threshold
                self.fireworthy = np.array([False] * (self.N ** 2))
                self.fireworthy[np.where(self.astates >= self.threshold)] = True
                self.astates = np.zeros(self.N ** 2)

                iters -= 1

            my_set_f = set(firing)
            comm_fired = 0 #number of communities where all nodes fired
            for c in self.communities:
                my_set_comm = set(c)
                comm_f = my_set_f.intersection(my_set_comm) #communities and firing intersection
                if len(comm_f) == len(my_set_comm):
                    comm_fired += 1
    

            self.fitness = np.average(props) + comm_fired
        
        else:
            # Set all states as not firing
            self.fireworthy = np.array([True] * (self.N ** 2))

            props = []

            iters = 50 # measuring fitness after 50 iterations
            while iters > 0: #and len(np.where(self.fireworthy == True)[0]) > 0:
                # Update activity states
                self.fire(iters)
                #print("fired")

                # Save all nodes that will fire
                firing = np.where(self.astates >= self.threshold)[0]

                # Find proportion of nodes that fire
                prop = len(firing) / self.N ** 2
                props.append(prop)

                # Reset firing states, keep those that are > threshold
                self.fireworthy = np.array([False] * (self.N ** 2))
                self.fireworthy[np.where(self.astates >= self.threshold)] = True
                self.astates = np.zeros(self.N ** 2)

                iters -= 1

            self.fitness = np.average(props)


    ### Visualization methods:

    def show_grid(self, size=10, labels=True):
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
        nx.draw(g, with_labels=labels, pos=pos, node_size=size)
        plt.show()

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
    return communities

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

'''
    def fire(self, node):
        """
        Fires a specific node

        Parameters:
            :int node:          The node being fired.

        Returns:
            None
        """
        neigh = list(self.adjL[node])

        if (len(neigh) > 0):
            self.astates[neigh] += self.fireweight
            
def spike(self):
        """
        Fires every node in the graph.

        Parameters:
            None

        Returns:
            None
        """

        # For each item being fired
        firing = np.where(self.fireworthy == True)[0]
        for i in firing:

            # Find all neighbors and increment activity state += 20
            neigh = list(self.adjL[i])

            if len(neigh) > 0:
                self.astates[neigh] += self.fireweight   
''' 