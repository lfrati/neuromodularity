# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% {"Collapsed": "false"}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time
import math
import json
import random

from scipy.stats import norm
from collections import Counter
from collections import OrderedDict, defaultdict

import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange

sns.set(style="white")


# %% [markdown] {"Collapsed": "false"}
# # Gaussian weights

# %% [markdown] {"Collapsed": "false"}
# Instead of having a separate copy of weights for each node we now have a "master_weights" matrix and each node is going to use a subset of that matrix prova prova prova

# %% {"Collapsed": "false"}
def id_to_coords(node, N):
    return node // N, node % N


# %% {"Collapsed": "false"}
def coords_to_id(i,j,N):
    return N*i + j


# %% {"Collapsed": "false"}
def test_coords_ids(N):
    for ID in range(N**2):
        coords = id_to_coords(ID,N)
        _ID = coords_to_id(*coords,N)
        assert _ID == ID, "IDs not matching {_ID} != {ID} for N={N}, coords=({coords})"
    print("Test ID -> coords -> ID: OK")
    
test_coords_ids(1000)


# %% {"Collapsed": "false"}
def make_master_weights(N, locality):
    rv = norm(loc=0, scale=locality)
    x = np.linspace(
        -1, 1, N * 2 + 1
    )  # size is twice as big because we are going to use NxN subsets of it
    p = rv.pdf(x)  # make marginal gaussians
    X, Y = np.meshgrid(
        p, p
    )  # use numpy magic to make them of the right shapes before combining them
    w = X * Y  # compute the 2D gaussian by multiplying the marginals together
    w /= w.sum()
    return w


# %% [markdown] {"Collapsed": "false"}
# Let's test this "master weights" thing to check it works

# %% {"Collapsed": "false"}
N = 30
locality = 0.2
W = make_master_weights(N=N, locality=locality)
plt.imshow(W)


# %% [markdown] {"Collapsed": "false"}
# So far so good, now let's take subsets of each for each node.
# We will see the gaussian "pimple" move around as the target node changes

# %% {"Collapsed": "false"}
def get_subset(W, node, N):
    assert 0 <= node and node < N * N, f"Node index out of bounds: "
    i, j = id_to_coords(node, N)  # convert node index to row,col coordinates
    subset = W[
        N - i : N - i + N, N - j : N - j + N
    ]  # extract a subset of the master_weights
    return np.copy(
        subset
    )  # make a copy to make sure subsequent manipulations don't affect the master


# %% {"Collapsed": "false"}
for node in np.arange(0, N * N, 200):
    subset = get_subset(W, node, N)
    plt.imshow(subset)
    plt.title(node)
    plt.show()


# %% [markdown] {"Collapsed": "false"}
# Ok, the weights look good. Before we can use them to sample from an hypothetical node we need to remove existing edges.
# We are going to make dummy adjacency list to simulate this scenario.

# %% {"Collapsed": "false"}
def compute_gaussian_weights(W, node, adjL):
    tmp, N = W.shape
    tmp, N = (
        tmp // 2,
        N // 2,
    )  # recover side-len from the weigths matrix, yeah, I did't want to have an extra parameter going around
    assert tmp == N, f"Weights have not the expected shape: Expected (,), got (,)"
    gauss = get_subset(
        W, node, N
    )  # get the appropiate subset in the manner we have shown above
    i, j = id_to_coords(node, N)  # zero the node coords to avoid self loops
    gauss[i][j] = 0
    for neigh in adjL[node]:  # go through the neighs in the adjlist and zero them
        i, j = id_to_coords(neigh, N)
        gauss[i][j] = 0
    gauss = (
        gauss / gauss.sum()
    )  # normalize everything to make sure we have probabilities
    return gauss.flatten()  # flatten them to use with np.random.choice


# %% {"Collapsed": "false"}
node = 43
dummy_adjL = {
    43: set([2, 45, 39, 70, 100, 250])
}  # edge 43 is connected to some random nodes
dummy_adjL

# %% {"Collapsed": "false"}
weights = compute_gaussian_weights(W, node, dummy_adjL)
plt.imshow(weights.reshape((N, N)))  # notice the black dots where edges exist already

# %% [markdown] {"Collapsed": "false"}
# # Sampling

# %% [markdown] {"Collapsed": "false"}
# We are now ready to use the weights to sample new edges

# %% {"Collapsed": "false"}
# Let's redefine our paramters to have everything in one place
N = 30
node = N * (N - 1) // 2  # pick a node in the center
locality = 0.2
W = make_master_weights(N=N, locality=locality)
dummy_adjL = {
    node: set([node - 2 * N, node + 2 * N, node + 2, node - 2])
}  # add edges in a cross pattern
weights = compute_gaussian_weights(W, node, dummy_adjL)
num_points = N ** 2
nodes = np.arange(N * N)

# %% [markdown] {"Collapsed": "false"}
# Test single samples, we should see a noisy gaussian structure

# %% {"Collapsed": "false"}
nodes = np.arange(N * N)
samples = [
    np.random.choice(nodes, p=weights, replace=False, size=1)[0] for _ in range(10000)
]

# How many times each node has been picked?
samples = Counter(samples).most_common()

results = np.zeros((N, N))
for coord, count in samples:
    # turn nodes id into coordinates for plotting
    x, y = coord // N, coord % N
    results[x][y] = count
plt.imshow(results)
plt.colorbar()

# %% [markdown] {"Collapsed": "false"}
# We can see that patern of empty dots around the center node we are sampling from. The method seems to work.

# %% [markdown] {"Collapsed": "false"}
# # Communicating agents

# %% [markdown] {"Collapsed": "false"}
# ## Utils

# %% {"Collapsed": "false"}
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
            #print()
    return communities

def make_layout(side_len):
    points = side_len ** 2
    x, y = np.meshgrid(np.arange(side_len), np.arange(side_len))  # make grid
    x, y = x.reshape(points), y.reshape(points)  # flatten to turn into pairs
    layout = {
        (points - 1 - idx): coords for idx, coords in enumerate(zip(reversed(x), y))
    }  # assign x,y to each node idx
    # points -1 and reverserd are used to match row/cols matrix format
    return layout

def get_neighbours(x,y,N):
    # return Moore neighbourhood
    return  [(x+i,y+j) 
               for j in range(-1,2) 
               for i in range(-1,2) 
               if x+i >= 0 and x+i < N
                and y+j >= 0 and y+j < N
                and (i != 0 or j !=0)]

def unzip(l):
    return list(zip(*l))


# %% [markdown] {"Collapsed": "false"}
# ## Network and Nodes

# %% {"Collapsed": "false"}
class Network:
    def __init__(self, comm_side, comms_per_side, locality, threshold):
        # REMEMBER WE ARE USING A 2D LATTICE
        # side length of each community
        self.comm_side = comm_side
        # number of nodes per community
        self.comm_size = self.comm_side * self.comm_side
        # number of communities per side
        self.comms_per_side = comms_per_side
        # number of nodes per side of the network
        self.N = self.comm_side * self.comms_per_side
        # total number of nodes in the network
        self.numNodes = self.N * self.N
        # list of lists of node IDs for each community
        self.communityIDs = make_communities(self.comm_side, self.comms_per_side)
        self.locality = locality
        # the master weights used to compute the gaussian weioght sampling
        self.W = make_master_weights(self.N, self.locality)
        # the weights used while rewiring, lower locality to promote connections between communities
        self.W_rewire = make_master_weights(self.N, self.locality*2)
        
        # list of nodes IDS i.e. [0,1,2,3,...,N*N]
        self.nodeIDs = np.arange(self.numNodes)
        self.threshold=threshold
        
        # TO BE INITIALIZED
        # list of the actual Nodes
        self.nodes = []
        # list of list of Nodes in each community
        self.communities = []
        self.adjL = {}
        
        self.activity_history = []
        self.spiking_history = []
        self.current_activity = []
        self.current_spiking = []
    
    def initialize(self,numEdges):
        self.nodes = [Node(ID, self) for ID in self.nodeIDs]
        self.communities = [[self.nodes[idx] for idx in ids] for ids in self.communityIDs]
        for commID, comm in enumerate(self.communities):
            for node in comm:
                node.community = commID
        self.adjL = {node:set() for node in self.nodeIDs}
        for _fromID in range(self.numNodes):
            weights = compute_gaussian_weights(self.W, _fromID, self.adjL)
            samples = np.random.choice(self.nodeIDs, p=weights, replace=False, size=numEdges)
            for _toID in samples:
                _from = self.nodes[_fromID]
                _to = self.nodes[_toID]
                self.adjL[_fromID].add(_toID)
                _from._out.add(_to)
                _to._in.add(_from)
        self.current_activity = np.zeros((self.N,self.N))
        self.current_spiking = np.zeros((self.N,self.N))
    
    def sample(self,node, gossip=None, eager=True):
        weights = compute_gaussian_weights(self.W, node, self.adjL)
        if gossip is not None:
            if eager:
                # look for node spiking a lot
                gossip /= gossip.sum()
            else:
                # look for weak nodes to help
                gossip = gossip.max() - gossip
                gossip /= gossip.sum()
                
            weights *= gossip.flatten()
            weights /= weights.sum()
        sample = np.random.choice(self.nodeIDs, p=weights, replace=False, size=1)[0]
        return sample
                
    def to_networkx(self):
        edges = {node:list(edges) for node,edges in self.adjL.items()}
        return nx.DiGraph(edges)
    
    def show(self):
        g = self.to_networkx()
        layout = make_layout(self.N)
        plt.figure(figsize=(16, 16))
        nx.draw(g, with_labels=False, pos=layout, node_size=80)
        
    def inspect(self):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        
        activity = axs[0].imshow(self.activity_history[-1])
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title("activity")
        # xgrid, ygrid = np.meshgrid(ticks, ticks)
        # axs[0].scatter(xgrid,ygrid, alpha=0.4)
        
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(activity, cax=cax)
        
        axs[1].imshow(self.spiking_history[-1])
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_title("spikes")
        plt.tight_layout(h_pad=1)
        plt.show()
        
    def spark(self):
        # reset all nodes' activity to zero, shut off spiking neurons
        [node.reset() for node in self.nodes]
        # select community and spark it's nodes
        community = random.choice(net.communities)
        [node.spark() for node in community]
    
    def check(self):
        # Check which nodes will spike.
        # A spike send signal to neighbours and reset the activity to zero.
        # Needs a separate phase to avoid setting the activity to zero before
        # all signals have been sent
        [node.check() for node in self.nodes]
    
    def gossip(self, sample=True):
        if(sample):
            # partial gossipping
            [node.diffuse() for node in random.sample(self.nodes, k=self.comm_size//2)]
        else:
            # Full gossip, for the brave of computing
            [node.diffuse() for node in random.sample(self.nodes, k=len(self.nodes))]
        
    def update(self):
        [node.update() for node in self.nodes]
        self.activity_history.append(self.current_activity)
        self.spiking_history.append(self.current_spiking)
        self.current_activity = np.zeros((self.N,self.N))
        self.current_spiking = np.zeros((self.N,self.N))
    
    def dead(self):
        return self.spiking_history[-1].sum() == 0
    
    def optimize(self):
        num_updates = np.random.binomial(net.numNodes,p=0.02)
        for _ in range(num_updates):
            winner = np.random.choice(self.nodes)
            new_edge = self.sample(winner.id, winner.info)
            old_edge = random.sample(winner._out,k=1)[0].id
            self.rewire(winner.id, old_edge, new_edge)
    
    def rewire(self,node:int,old:int,new:int):
        assert old in self.adjL[node], f"Rewiring non-existing wire {node} -> {old}"
        assert new not in self.adjL[node], f"Rewiring existing wire {node} -> {new}"
        
        # remove from adjL
        self.adjL[node].discard(old)
        # add new to adjL
        self.adjL[node].add(new)
        
        # get nodes from IDs
        node = self.nodes[node]
        old = self.nodes[old]
        new = self.nodes[new]
        
        # remove node from node.out[old].in
        new._in.discard(node)
        # remove old from node.out
        node._out.discard(old)
        
        # add new to node.out
        node._out.add(new)
        # add node to new.in
        new._in.add(node)
        
    def mu(self):
        within = 0
        tot = 0
        for node in self.nodes:
            for to in node._out:
                if to.community == node.community:
                    within += 1
                tot += 1
        return within/tot


# %% {"Collapsed": "false"}
class Node:
    def __init__(self, ID, network):
        self.id = ID
        self.N = network.N
        self.coords = id_to_coords(self.id, self.N)
        self.threshold = network.threshold
        self.activity = 0
        self._out = set()
        self._in = set()
        self.spiking = False
        self.network = network
        # current info, sent to neighbours while diffusing
        self.info = np.zeros((self.N,self.N))
        # updated info, use separate board to avoid diffusing information we just got
        self.new_info = np.zeros_like(self.info)
        
        self.neighbourhood = get_neighbours(*self.coords, self.N)
        
    def excite(self):
        # increse node activity and record it in history
        self.network.current_activity[self.coords] += 1
        self.activity += 1
    
    def spark(self):
        # force node to spike, used by the network to start activity
        self.spiking = True
        self.activity = self.threshold
    
    def check(self):
        # check if the node will spike. Needs a separate phase
        # because activity will reset to zero
        if self.activity >= self.threshold:
            self.spiking = True
            self.network.current_spiking[self.coords] = 1
            # update both boards to avoid loosing the new info added
            # (info is sent around but new_info is used to update info)
            self.info[self.coords] += 1
            self.new_info[self.coords] += 1
        self.activity = 0
    
    def gossip(self,new_info):
        self.new_info = np.maximum(self.new_info, new_info)
            
    def update(self):
        # if spiking signal neighbouring nodes
        if self.spiking:
            [neigh.excite() for neigh in self._out]
        self.spiking = False
        self.info = np.copy(self.new_info)
    
    def diffuse(self):
        [self.network.nodes[coords_to_id(*neigh,net.N)].gossip(self.info) for neigh in self.neighbourhood]
    
    def reset(self):
        self.spiking = False
        self.activity = 0
        # TODO: RESET GOSSIP?
        
    def __repr__(self):
        return f"ID: {self.id}"


# %% {"Collapsed": "false"}

# %% [markdown] {"Collapsed": "false"}
# ## Testing

# %% {"Collapsed": "false"}
comm_side = 4
comms_per_side = 5
threshold = int((comm_side*comm_side - 1)/2)
wires = comm_side*comm_side - 1
net = Network(comm_side=comm_side,comms_per_side=comms_per_side,locality=0.1, threshold=threshold)
print(f"wires {wires}, threshold {threshold}")

# %% {"Collapsed": "false"}
net.initialize(wires)

# %% {"Collapsed": "false"}

# %% {"Collapsed": "false"}
net.spark()
net.check()
net.gossip(sample=False)
net.update()
net.inspect()

# %% {"Collapsed": "false"}
net.check()
net.gossip(sample=False)
net.update()
net.inspect()

# %% {"Collapsed": "false"}
net.check()
net.gossip(sample=False)
net.update()
net.inspect()

# %% {"Collapsed": "false"}
pre_edges = sum([len(l) for l in net.adjL.values()])
pre_edges

# %% {"Collapsed": "false"}
net.optimize()

# %% {"Collapsed": "false"}

# %% {"Collapsed": "false"}
metrics = []

# %% {"Collapsed": "false", "jupyter": {"outputs_hidden": true}}
t = trange(100)
for it in t:
    net.spark()
    t.set_description(f"ITERATION {it}")
    for ep in range(10):
        net.check()
        net.gossip()
        net.update()
        #net.inspect()
        if(net.dead()):
            break
    net.optimize()
    metrics.append((ep, net.mu()))

# %% {"Collapsed": "false"}
survival,mus = unzip(metrics)
plt.hist(survival)
plt.xlabel("time of death")
plt.ylabel("num of trials")

# %% {"Collapsed": "false"}
plt.plot(mus)

# %% {"Collapsed": "false"}
# To check how well gossipping is doing I'll compute the activity ground-truth
gt = np.array(net.spiking_history).sum(axis=0)
gt /= gt.max() # we are interested in proportions of activity, not actual value
plt.imshow(gt)
plt.title("Spiking Ground Truth")
plt.colorbar()

# %% {"Collapsed": "false"}
# gather the info shared by all agents by gossiping
collective_info = np.array([node.info for node in net.nodes])
collective_info = collective_info.sum(axis=0)
        
collective_info /= collective_info.max()
plt.imshow(collective_info)
plt.title("Average gossip info")
plt.colorbar()

# %% {"Collapsed": "false"}
# compare the two by the abs diff of each node against the gt
# gossip status -> (gt-gossip)
gossip = np.zeros((net.N,net.N))
for node in net.nodes:
    gossip_matrix = np.copy(node.info)
    # normalize gossip matrix if there is any gossip
    gossip_level = node.info.max()
    if(gossip_level > 0):
        gossip_matrix /= gossip_level
    perf = np.abs(gt-gossip_matrix).sum()
    gossip[node.coords] = perf

gossip /= gossip.max()

plt.imshow(gossip)
plt.colorbar()
plt.title("Abs(gt-gossip) per node")
plt.show()

# %% {"Collapsed": "false"}
# check agents agreement using the std of their gossip "matrices"
# compare the previous gt with the info shared by all agents by gossiping
collective_info = np.array([node.info for node in net.nodes])
        
plt.imshow(collective_info.std(axis=0))
plt.title("Gossip std. per node")
plt.colorbar()

# %% {"Collapsed": "false"}
# Check moore neighbour
ID = coords_to_id(0,17,net.N)
print(net.nodes[ID].coords)
neighs = net.nodes[ID].neighbourhood
board = np.zeros((net.N,net.N))
for x,y in neighs:
    board[x][y] = 1
plt.imshow(board)

# %% {"Collapsed": "false"}
tmp = np.copy(net.nodes[ID].info)
#tmp[id_to_coords(ID,net.N)] = 2
plt.imshow(tmp)
plt.grid()
plt.colorbar()

# %% {"Collapsed": "false"}
gossip = net.nodes[25].info
gossip = gossip.max() - gossip
plt.imshow(gossip)
plt.colorbar()

# %% {"Collapsed": "false"}
weights = get_subset(net.W_rewire, 0, net.N)

#weights *= gossip
#weights /= weights.sum()

fig,ax = plt.subplots(figsize=(10,10))

activity = plt.imshow(weights)

ticks = list(range(net.N))
plt.xticks(ticks)
plt.yticks(ticks)
xgrid, ygrid = np.meshgrid(ticks, ticks)
plt.scatter(xgrid,ygrid, s=10,alpha=0.4)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(activity, cax=cax)

plt.show()

# %% {"Collapsed": "false"}
comms = np.zeros((net.N,net.N))
for node in net.nodes:
    comms[node.coords] = node.community
plt.imshow(comms)

# %% {"Collapsed": "false"}
net.show()

# %% {"Collapsed": "false"}
n = 100
params = np.zeros((n,n))
for comm_side in range(n):
    for comms_per_side in range(n):
        wires = comm_side*(comm_side-1)*(comms_per_side**2)
        params[comm_side][comms_per_side]=wires

plt.imshow(params)
plt.ylabel("comm_side")
plt.xlabel("comms_per_side")
plt.colorbar()

# %% {"Collapsed": "false"}
c = 10
plt.plot(params[c,:],label=f"comm_side={c}")
plt.plot(params[:,c],label=f"comms_per_side={c}")
plt.xlabel("n")
plt.ylabel("edges")
plt.legend()

# %% {"Collapsed": "false"}

# %% {"Collapsed": "false"}

# %% {"Collapsed": "false"}
