from network import Network
from utils import unzip
from tqdm import trange
import matplotlib.pyplot as plt

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
print(f"wires {wires}, threshold {threshold}")

net.initialize(wires)
metrics = []

t = trange(100)
for it in t:
    net.spark()
    t.set_description(f"ITERATION {it}")
    for ep in range(10):
        net.check()
        net.gossip()
        net.update()
        # net.inspect()
        if net.dead():
            break
    net.optimize()
    metrics.append((ep, net.mu()))

survival, mus = unzip(metrics)
plt.hist(survival)
plt.xlabel("time of death")
plt.ylabel("num of trials")
