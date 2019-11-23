import numpy as np
import random
from collections import OrderedDict, defaultdict
from collections import Counter
from scipy.stats import norm
import pandas as pd
from copy import deepcopy
import time
import json

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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


class Network:
    def __init__(self, N, opts, backup=False):
        self.N = N
        self.adjM = np.zeros((N ** 2, N ** 2))
        self.adjL = defaultdict(set)
        self.nodes = np.arange(N ** 2)
        self.is_backup = backup
        self.opts = opts

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

        if backup == False:
            self.backup = Network(N, opts, backup=True)

    def add_edges(self, node, num_samples):
        num_samples = max(num_samples, self.N - 1 - len(self.backup.adjL[node]))
        samples = np.random.choice(
            self.nodes, p=self.weights[node], replace=False, size=num_samples
        )
        for sample in samples:
            self.weights[node][sample] = 0
            self.adjM[node][sample] = 1
            self.adjL[node].add(sample)

        self.weights[node] /= self.weights[node].sum()

        return samples

    def undo(self):
        self.adjM = np.copy(self.backup.adjM)
        self.adjL = deepcopy(self.backup.adjL)
        self.weights = np.copy(self.backup.weights)

    def commit(self):
        self.backup.adjM = np.copy(self.adjM)
        self.backup.adjL = deepcopy(self.adjL)
        self.backup.weights = np.copy(self.weights)

    def show_adj(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.adjM)
        plt.show()

    def serialize(self):
        opts = deepcopy(self.opts)
        opts["N"] = self.N
        info = [json.dumps(opts)+"\n"]
        timestr = time.strftime("%Y-%m-%d_%H%M%S")
        name = timestr + "_" + "_".join(
            [name + "_" + str(val).replace(".", "") for name, val in self.opts.items()]
        )
        name += ".edgelist"
        data = ["{} {}\n".format(node,neigh) for node,neighs in self.adjL.items() for neigh in neighs]
        with open(name,'w') as f:
            f.writelines(info + data)
        return name
