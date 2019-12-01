import numpy as np 
import pandas as pd 

from UniGaussianNets import *

class Population:
    """
    This class acts as a generic container for multiple 'inviduals' (i.e.
    multiple networks).

    This will take advantage of a basic mutation / selection scheme.
    """

    def __init__(self, popsize, indsize, opts, threshold):
        """
        Initializes a list of networks of a given size, distribution,
        and locality. 

        Parameters:
            :int size:      Number of inviduals in the population.
            :int indsize:   Size of networks.
            :dict opts:     Network specific options.
            :int threshold: The threshold of the networks.

        Returns:
            None. Costructor method.
        """

        self.opts = opts
        self.size = popsize
        self.indsize = indsize
        self.population = [Network(indsize, opts, threshold) 
        for _ in range(popsize)]

    def initialize(self):
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
            i.set_up() #adds specific number of edges rather than using average degree
            
    def initialize_unif(self): #specific number of edges uniformly distributed
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
            i.set_up_unif()

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
            i.evaluate_1comp() #here the fitness only has 1 component 