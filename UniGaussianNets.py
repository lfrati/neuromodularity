import time
import math
import json
import random
from copy import deepcopy

from scipy.stats import norm
from collections import Counter
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import networkx as nx


# This import registers the 3D projection, but is otherwise unused.
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

### HELPER FUNCTIONS
def make_communities(community_side, communities_per_side): #this is for creating communities but it is not used in this version of the code
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
            print()
    return communities


def gauss(x, mean, scale):  # values here were tweaked for our fitness function
    return (
        1
        / (scale * math.sqrt(2 * math.pi))
        * (math.e ** -(0.5 * (x - mean) ** 2 / scale ** 2))
    ) - 0.015

    y = np.linspace(1, 100, 100)
    x = gauss(y, 70, 20)
    plt.plot(x)


def node_to_idxs(node, N):
    return [node // N, node % N]


def unzip(l):
    return list(zip(*l))


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


def show_weights(weights, N):
    i = 0
    while i < len(weights):
        w = weights[i].reshape(N, N)
        plt.imshow(w)
        plt.show()
        i += np.random.randint(N ** 2 // 3)


def show_3d_weights(weights, N):
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Make data.
    X = np.arange(0, N)
    Y = np.arange(0, N)
    X, Y = np.meshgrid(X, Y)
    Z = weights.reshape(N, N)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

    # Customize the z axis.
    # a x.set_zlim(0, .001)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


### MAIN CLASS


class Network:
    def __init__(
        self, N=5, opts={"distr": "gauss", "locality": 0.2}, threshold=100, ID=0
    ):
        """
        Initialization of network object.

        Parameters:
            :int N:         The side length of the network grid.
            :dict opts:     The dictionary of parameters.

        Returns:
            None. Constructor method.
        """

        # Define default attributes
        self.N = N
        self.opts = opts
        self.age = 0
        self.muts = 0
        self.adjM = np.zeros((N ** 2, N ** 2))
        self.adjL = defaultdict(set)
        self.nodes = np.arange(N ** 2)
        self.fitness = 0
        self.threshold = threshold

        self.astates = np.zeros(N ** 2)
        self.fireworthy = np.array([False] * (N ** 2))

        # Define more default attributes
        if opts["distr"] == "gauss":
            self.distr = opts["distr"]
            self.locality = opts["locality"]
            marginals = normal_marginals(self.N, self.locality)
            self.weights = gen_gauss_weights(marginals)

        elif opts["distr"] == "unif":
            self.distr = opts["distr"]
            self.weights = gen_unif_weights(self.N)
        else:
            raise RuntimeError("Distribution not implemented: " + opts["distr"])

    def add_edges(self, node, num_samples):
        """
        Adds a number of edges to a given node.

        Parameters:
            :int node:          The node that is being connected to (source)
            :int num_samples:   The number of nodes to be connected to the source

        Returns:
            None. Changes adjL and adjM.
        """

        # Verify that it is possible to sample enough nodes.
        num_samples = min(num_samples, self.N - 1 - len(self.adjL[node]))

        # Find target nodes as sample.
        samples = np.random.choice(
            self.nodes, p=self.weights[node], replace=False, size=num_samples
        )

        # Connect to source node
        for sample in samples:
            self.weights[node][sample] = 0
            self.adjM[node][sample] = 1
            self.adjL[node].add(sample)

        # Set weights
        self.weights[node] /= self.weights[node].sum()

        return samples

    def add_edges_unif(self, node, num_samples): #adding edges uniformly
        """
        Adds a number of edges to a given node.

        Parameters:
            :int node:          The node that is being connected to (source)
            :int num_samples:   The number of nodes to be connected to the source

        Returns:
            None. Changes adjL and adjM.
        """

        # Verify that it is possible to sample enough nodes.
        num_samples = min(num_samples, self.N - 1 - len(self.adjL[node]))

        # Find target nodes as sample.
        availableNodes = list(set(self.nodes) - self.adjL[node])
        samples = np.random.choice(availableNodes, replace=False, size=num_samples)

        # Connect to source node
        for sample in samples:
            self.adjM[node][sample] = 1
            self.adjL[node].add(sample)

        return samples

    def populate(self, av_k):
        """
        Adds edges to the initialised empty network.

        Parameters:
            :N:                   Number of nodes in the network.
            :av_k:                Average degree.

        Returns:
            None (adds edges to empty network).
        """

        for node in range(self.N ** 2):
            new_edges = np.random.normal(loc=av_k, scale=1.0)
            if new_edges < 0:
                new_edges = 0

            self.add_edges(node, int(new_edges))

    def populate_unif(self, av_k): #initialise edges uniformly
        """
        Adds edges to the initialised empty network.

        Parameters:
            :N:                   Number of nodes in the network.
            :av_k:                Average degree.

        Returns:
            None (adds edges to empty network).
        """

        for node in range(self.N ** 2):
            new_edges = np.random.normal(loc=av_k, scale=1.0)
            if new_edges < 0:
                new_edges = 0

            self.add_edges_unif(node, int(new_edges))

    def mutate(self):
        """
        A simple wrapper for add_edges: It chooses a random node and 
        performs a sample on it. Used in evolution.

        Parameters:
            None

        Returns:
            None
        """

        # Add a single edge to a random node, increment age
        node = random.randrange(0, self.N ** 2)
        self.add_edges(node, 1) #when growing edges uniformly use self.add_edges_unif()
        self.muts += 1

    def fire(self):
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
                self.astates[neigh] += 20

    def evaluate(self):
        """
        Evaluates the performance of a network on two main pieces

        Parameters:
            None

        Returns:
            None
        """

        ### Find first fitness component: ###
        # Making sure one edge does not dominate fire count

        edges = []
        props = []

        # Set all states as firing
        self.fireworthy = np.array([True] * (self.N ** 2))

        # While there are still SOME fireworthy states:
        iters = 1000
        while iters > 0 and len(np.where(self.fireworthy == True)[0]) > 0:
            # Update activity states
            self.fire()
            print("fired")

            # Save all edges associated with fireworthy nodes
            firing = np.where(self.astates >= self.threshold)[0]

            # Find proportion of nodes that fire
            prop = len(firing) / self.N ** 2
            props.append(
                gauss((prop * 100), 70, 15)
            )  # this is the fitness component for overall node activity

            for i in firing:
                edge = [(i, x) for x in self.adjL[i]]  # this tracks edge activity
                edges += edge

            # Reset firing states, keep those that are > threshold
            self.fireworthy = np.array([False] * (self.N ** 2))
            self.fireworthy[np.where(self.astates >= self.threshold)] = True
            self.astates = np.zeros(self.N ** 2)

            iters -= 1

        num_edges = sum(map(len, self.adjL.values()))

        if len(edges) > 0:
            ecount = sorted(Counter(map(tuple, edges)).values(), reverse=True)
            T = sum(ecount) * 0.75
            summ = 0
            idx = 0
            while summ <= T:
                summ = summ + ecount[idx]
                idx += 1
        else:
            idx = 0

        ### Find first component of fitness ###
        fitness_comp_1 = ((idx / num_edges) * 100) / 2  # this goes from 0 to 75/2
        ###                                 ###
        print(fitness_comp_1)

        ### Find second component of fitness ###
        fitness_comp_2 = (
            (np.average(props) * 1000) + 15
        ) * 4  # fitness 2 is more important, 0 to 80
        ###                                  ###
        print(fitness_comp_2)

        # Set fitness to SUM
        self.fitness = np.sum([fitness_comp_1, fitness_comp_2])

    def show_adj(self):
        """
        Visualizes adjacency matrix.
        """

        plt.figure(figsize=(10, 10))
        plt.imshow(self.adjM)
        plt.show()

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

    def serialize(self):
        """
        Saves the attributes of a given network.
        """

        # Find some init params.
        opts = deepcopy(self.opts)
        opts["N"] = self.N

        # Dump them all into a JSON, save with date/time
        info = [json.dumps(opts) + "\n"]
        timestr = time.strftime("%Y-%m-%d_%H%M%S")
        name = (
            timestr
            + "_"
            + "_".join(
                [
                    name + "_" + str(val).replace(".", "")
                    for name, val in self.opts.items()
                ]
            )
        )
        name += ".edgelist"
        data = [
            "{} {}\n".format(node, neigh)
            for node, neighs in self.adjL.items()
            for neigh in neighs
        ]
        with open(name, "w") as f:
            f.writelines(info + data)
        return name
