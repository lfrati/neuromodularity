from utils import make_communities, make_master_weights, compute_gaussian_weights, make_layout, id_to_coords
from node import Node
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx


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
        self.W_rewire = make_master_weights(self.N, self.locality * 2)

        # list of nodes IDS i.e. [0,1,2,3,...,N*N]
        self.nodeIDs = np.arange(self.numNodes)
        self.threshold = threshold

        # TO BE INITIALIZED
        # list of the actual Nodes
        self.nodes = []
        # list of list of Nodes in each community
        self.communities = []
        self.adjL_in = {}
        self.adjL_out = {}

        self.activity_history = []
        self.spiking_history = []
        self.current_activity = []
        self.current_spiking = []

    def cheat(self):
        self.nodes = [Node(ID, self) for ID in self.nodeIDs]
        self.communities = [
            [self.nodes[idx] for idx in ids] for ids in self.communityIDs
        ]
        for commID, comm in enumerate(self.communities):
            for node in comm:
                node.community = commID
        self.adjL_in = {node: set() for node in self.nodeIDs}
        self.adjL_out = {node: set() for node in self.nodeIDs}
        for _from in self.nodes:
            _fromID = _from.id
            samples = [neigh for neigh in self.communityIDs[_from.community] if neigh != _fromID]
            for _toID in samples:
                self.adjL_in[_toID].add(_fromID)
                self.adjL_out[_fromID].add(_toID)
                
        self.current_activity = np.zeros((self.N, self.N))
        self.current_spiking = np.zeros((self.N, self.N))

    def initialize(self, numEdges):
        self.nodes = [Node(ID, self) for ID in self.nodeIDs]
        self.communities = [
            [self.nodes[idx] for idx in ids] for ids in self.communityIDs
        ]
        for commID, comm in enumerate(self.communities):
            for node in comm:
                node.community = commID
        self.adjL_in = {node: set() for node in self.nodeIDs}
        self.adjL_out = {node: set() for node in self.nodeIDs}
        for _fromID in range(self.numNodes):
            weights = compute_gaussian_weights(self.W, _fromID, self.adjL_out)
            samples = np.random.choice(
                self.nodeIDs, p=weights, replace=False, size=numEdges
            )
            for _toID in samples:
                self.adjL_in[_toID].add(_fromID)
                self.adjL_out[_fromID].add(_toID)
        self.current_activity = np.zeros((self.N, self.N))
        self.current_spiking = np.zeros((self.N, self.N))

    def sample(self, node, gossip=None, firing=False):
        # if the node is firing use gossip info to find who to help
        # if the node is not firing use gaussian weights
        if firing and gossip is not None and gossip.sum() > 0:
            # look for weak nodes to help
            weights = np.copy(gossip.max() - gossip)
            weights = weights.flatten()
            for neigh in self.adjL_out[node]:  # go through the neighs in the adjlist and zero them
                weights[neigh] = 0
            
            # we could have zeroed the only non-zero terms so we need to check again
            if weights.sum() > 0:
                fat_gauss = compute_gaussian_weights(self.W_rewire, node, self.adjL_out)
                weights *= fat_gauss
                weights /= weights.sum()
            else:
                weights = compute_gaussian_weights(self.W, node, self.adjL_out)
        else:
            weights = compute_gaussian_weights(self.W, node, self.adjL_out)

        sample = np.random.choice(self.nodeIDs, p=weights, replace=False, size=1)[0]

        return sample

    def to_networkx(self):
        edges = {node: list(edges) for node, edges in self.adjL_out.items()}
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
        community = random.choice(self.communities)
        [node.spark() for node in community]

    def check(self):
        # Check which nodes will spike.
        # A spike send signal to neighbours and reset the activity to zero.
        # Needs a separate phase to avoid setting the activity to zero before
        # all signals have been sent
        [node.check() for node in self.nodes]

    def gossip(self, sample=True):
        if sample:
            # partial gossipping
            [
                node.diffuse()
                for node in random.sample(self.nodes, k=self.comm_size // 2)
            ]
        else:
            # Full gossip, for the brave of computing
            [node.diffuse() for node in random.sample(self.nodes, k=len(self.nodes))]

    def update(self):
        [node.update() for node in self.nodes]
        self.activity_history.append(self.current_activity)
        self.spiking_history.append(self.current_spiking)
        self.current_activity = np.zeros((self.N, self.N))
        self.current_spiking = np.zeros((self.N, self.N))

    def dead(self):
        return self.spiking_history[-1].sum() == 0

    def optimize(self):
        num_updates = np.random.binomial(self.numNodes, p=0.2)
        for _ in range(num_updates):
            # pick a node to optimze
            node = np.random.choice(self.nodes)
            # we either rewire or trade edges, so we need at least one
            if len(self.adjL_out[node.id]) <= 0:
                continue
            spiking = self.current_spiking[id_to_coords(node.id,self.N)] == 1
            gossip = node.info

            # if the node is spiking look for someone to help ( i.e. gossip.max() - gossip )
            if spiking and gossip.sum() > 0:
                # look for weak nodes to help
                weights = np.copy(gossip.max() - gossip)
        
            # if the node is NOT spiking look for popular nodes to have them help
            else:
                weights = np.copy(gossip).flatten()
                
            weights = weights.flatten()
            for neigh in self.adjL_out[node.id]:  # go through the neighs in the adjlist and zero them
                weights[neigh] = 0

            # we could have zeroed the only non-zero terms so we need to check again
            if weights.sum() > 0:
                fat_gauss = compute_gaussian_weights(self.W_rewire, node.id, self.adjL_out)
                tmp_weights = weights * fat_gauss # apply gaussian scaling, might zero everything
                if tmp_weights.sum() > 0:
                    weights = tmp_weights
                weights /= weights.sum() # normalize to prob
            else:
                # we have no gossip to leverage, use the fat gaussian
                weights = compute_gaussian_weights(self.W_rewire, node.id, self.adjL_out)
            
            # new edge to be added
            new_edge = np.random.choice(self.nodeIDs, p=weights, replace=False, size=1)[0]
            # old edge to be removed
            old_edge = random.sample(self.adjL_out[node.id], k=1)[0]

            if spiking:
                # we connect the current node to the new node (i.e. send signal)
                self.rewire(node.id, old_edge, node.id, new_edge)
            else:
                # we connect the new node to the current node (e.g. receive signal)
                self.rewire(node.id, old_edge, node.id, new_edge)
            
            

    def rewire(self, old_from: int, old_to: int, new_to: int, new_from: int):
        
        # update outgoing edges
        self.adjL_out[old_from].discard(old_to)
        self.adjL_out[new_from].add(new_to)

        # updated incoming edges
        self.adjL_in[new_to].add(new_from)
        self.adjL_in[old_to].discard(old_from)
        

    def mu(self):
        within = 0
        between = 0
        tot = 0
        for node in self.nodes:
            for to in self.adjL_out[node.id]:

                if self.nodes[to].community == node.community:
                    within += 1
                else:
                    between += 1
                tot += 1
        return between / tot
