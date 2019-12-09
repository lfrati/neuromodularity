import numpy as np
from scipy.stats import norm


def id_to_coords(node, N):
    return node // N, node % N


def coords_to_id(i, j, N):
    return N * i + j


def test_coords_ids(N):
    for ID in range(N ** 2):
        coords = id_to_coords(ID, N)
        _ID = coords_to_id(*coords, N)
        assert _ID == ID, "IDs not matching {_ID} != {ID} for N={N}, coords=({coords})"
    print("Test ID -> coords -> ID: OK")


test_coords_ids(1000)


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


def get_subset(W, node, N):
    assert 0 <= node and node < N * N, f"Node index out of bounds: "
    i, j = id_to_coords(node, N)  # convert node index to row,col coordinates
    subset = W[
        N - i : N - i + N, N - j : N - j + N
    ]  # extract a subset of the master_weights
    return np.copy(
        subset
    )  # make a copy to make sure subsequent manipulations don't affect the master


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
            # print()
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


def get_neighbours(x, y, N):
    # return Moore neighbourhood
    return [
        (x + i, y + j)
        for j in range(-1, 2)
        for i in range(-1, 2)
        if x + i >= 0 and x + i < N and y + j >= 0 and y + j < N and (i != 0 or j != 0)
    ]


def unzip(l):
    return list(zip(*l))
