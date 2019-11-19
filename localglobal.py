import random
import numpy as np 
import pandas as pd
import seaborn as sns
import networkx as nx
from scipy.stats import norm
import matplotlib.pyplot as plt

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

    def __init__(self, L = 10, scale = 1):
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
        self.adjlist[node].update([(node, to) for to in self.L 
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
        
        self.adjlist[node].update([(node, to) for to in edges])

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
        local_edges = [(node, to) for to in self.L * to_row + to_col if node != to]

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
        
        edges = []
        for i in self.adjlist.values():
            if (len(i) > 0):

                # Fund edges in a networkx friendly format.
                edges += list(i)

        # Fix position mapping, because networkx is screwy.
        pos = {x:(self.mapping[x][1], self.L - self.mapping[x][0])
              for x in self.mapping.keys()}

        g = nx.DiGraph()
        g.add_nodes_from(self.mapping.keys())
        g.add_edges_from(edges)
        plt.figure(figsize=(16, 16))
        nx.draw(g, with_labels=labels, pos=pos, node_size=size)
        plt.show()


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
