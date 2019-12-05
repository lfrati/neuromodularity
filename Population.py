import numpy as np 
import pandas as pd 

from Network import *



class Population:
    """
    This class acts as a generic container for multiple 'inviduals' (i.e.
    multiple networks).

    This will take advantage of a basic mutation / selection scheme.
    """

    def __init__(self, **kwargs):
        """
        Initializes a list of networks of a given size, distribution,
        and locality. 

        Parameters:
            :int size:          Number of inviduals in the population.
            :int indsize:       Size of networks.
            :int fireweight:    How strong a firing node is.
            :str stype:         The distribution of the networks
            :float locality:    The locality of the networks
            :int threshold:     The threshold of the networks.
            :tuple comshape:    Community shape. Com size * coms per side
            :list mweights:     Sampling matrix.

        Returns:
            None. Costructor method.
        """

        # Set some defaults
        kwargs.setdefault('popsize', 10)
        kwargs.setdefault('indsize', 10)
        kwargs.setdefault('stype', 'gaussian')
        kwargs.setdefault('locality', 0.25)
        kwargs.setdefault('threshold',100)
        kwargs.setdefault('comshape', None)
        kwargs.setdefault('fireweight', 20)

        self.__dict__.update(kwargs)

        kwargs.setdefault('mweights',make_master_weights(self.indsize, self.locality))

        self.__dict__.update(kwargs)

        self.population = [Network(
            N = self.indsize, 
            dist = self.stype, 
            locality = self.locality,
            threshold = self.threshold,
            comshape = self.comshape,
            fireweight = self.fireweight,
            mweights = self.mweights) 
        for _ in range(self.popsize)]

    def initialize(self, av_k):
        """
        Fills empty networks in population with initial connections.

        Parameters:
            :int av_k:          Average degree of each node.

        Returns:
            None.
        """

        # For each individual
        for i in self.population:

            # Initialize / populate them.
            i.populate(av_k)

    def mutate(self):
        """
        Population-wide wrapper for mutate function. Performs single edge
        addition to every network in the population.
        
        Parameters:
            None

        Returns:
            None
        """

        # For each individual in the population
        for i in self.population:

            # Mutate them
            i.mutate()

    def evaluate(self):
        """
        Population-wide wrapper for evaluation (fitness) function. Finds and
        assigns fitness for each individual of the population.

        Parameters:
            None

        Returns:
            None
        """

        # For each individual in the population:
        for i in self.population:

            # Evaluate them:
            i.evaluate()

### HELPER FUNCTIONS

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