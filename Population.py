import numpy as np 
import pandas as pd 

from UniGaussianNets import *

class Population:
    """
    This class acts as a generic container for multiple 'inviduals' (i.e.
    multiple networks).

    This will take advantage of a basic mutation / selection scheme.
    """

    def __init__(self, popsize, indsize, dist, locality, threshold, comshape):
        """
        Initializes a list of networks of a given size, distribution,
        and locality. 

        Parameters:
            :int size:          Number of inviduals in the population.
            :int indsize:       Size of networks.
            :str dist:          The distribution of the networks
            :float locality:    The locality of the networks
            :int threshold:     The threshold of the networks.
            :tuple comshape:    Community shape. Com size * coms per side

        Returns:
            None. Costructor method.
        """

        self.dist = dist 
        self.locality = locality 
        self.size = popsize
        self.indsize = indsize
        self.comshape = comshape
        self.population = [Network(
            N = self.indsize, 
            dist = self.dist, 
            locality = self.locality,
            threshold = threshold,
            comshape = comshape) 
        for _ in range(popsize)]

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