from utils import id_to_coords, get_neighbours, coords_to_id
import numpy as np


class Node:
    def __init__(self, ID, network):
        self.id = ID
        self.N = network.N
        self.coords = id_to_coords(self.id, self.N)
        self.threshold = network.threshold
        self.activity = 0
        self.spiking = False
        self.network = network
        # current info, sent to neighbours while diffusing
        self.info = np.zeros((self.N, self.N))
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

    def gossip(self, new_info):
        self.new_info = np.maximum(self.new_info, new_info)

    def update(self):
        # if spiking signal neighbouring nodes
        if self.spiking:
            [self.network.nodes[neigh].excite() for neigh in self.network.adjL_out[self.id]]
        self.spiking = False
        self.info = np.copy(self.new_info)

    def diffuse(self):
        [
            self.network.nodes[coords_to_id(*neigh, self.N)].gossip(self.info)
            for neigh in self.neighbourhood
        ]

    def reset(self):
        self.spiking = False
        self.activity = 0
        # TODO: RESET GOSSIP?

    def __repr__(self):
        return f"ID: {self.id}"
