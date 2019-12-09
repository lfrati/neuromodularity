from network import Network
from utils import unzip
from tqdm import trange
import matplotlib.pyplot as plt

import unittest

def make_net():
    comm_side = 4
    comms_per_side = 5
    threshold = int((comm_side * comm_side - 1) / 2)
    wires = comm_side * comm_side - 1
    net = Network(
        comm_side=comm_side,
        comms_per_side=comms_per_side,
        locality=0.1,
        threshold=threshold,
    )
    net.initialize(wires)
    return net

class TestNetworks(unittest.TestCase):
    def test_run(self):
        net = make_net()
        net.spark()
        net.check()
        net.gossip()
        net.update()
        net.optimize()

    def test_rewire_same_start(self):
        net = make_net()
        
        pre_edges = sum([len(l) for l in net.adjL.values()])
        old_from = net.nodes[0]
        old_to = list(net.nodes[0]._out)[0]
        new_to = net.nodes[-1]

        net.rewire(old_from.id, old_to.id, old_from.id, new_to.id)

        post_edges = sum([len(l) for l in net.adjL.values()])
        self.assertEqual(pre_edges, post_edges)

    def test_rewire_swap(self):
        net = make_net()
        
        pre_edges = sum([len(l) for l in net.adjL.values()])

        old_from = net.nodes[0]
        old_to = list(net.nodes[0]._out)[0]
        new_to = net.nodes[-1]

        net.rewire(old_from.id, old_to.id, new_to.id, old_from.id)
        
        post_edges = sum([len(l) for l in net.adjL.values()])
        self.assertEqual(pre_edges, post_edges)

if __name__ == '__main__':
    unittest.main()