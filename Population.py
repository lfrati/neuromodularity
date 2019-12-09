import numpy as np 
import pandas as pd 

from Network import *



class Population:
    """
    This class acts as a generic container for multiple 'inviduals' (i.e.
    multiple networks).

    This will take advantage of a basic mutation / selection scheme.
    """

    def __init__(self, net_type, popsize, locality,  **kwargs):
        """ 
        Creates networks of a given type based on key word arguments.
    
        Parameters:
            :str net_type:      Specifies the type of network in the pop.

        Returns:
            None. Costructor method.
        """
        self.net_type = net_type
        self.popsize = popsize
        self.locality = locality

        self.last_id = 0
        self.population = []

        self.__dict__.update(**kwargs)

        if (net_type == "no_com"):
            self.population = [NoCommunity(**kwargs) for i in range(popsize)]

        elif (net_type == 'gaussian_com'):
            try:
                # Create weights and pass to network.
                self.mweights = make_master_weights(self.com_side * self.coms_per_side,
                    self.locality)
                kwargs.update({'mweights':self.mweights, 'locality':self.locality})

                for i in range(self.popsize):
                    kwargs.update({'ID':self.last_id})
                    self.population.append(GaussianCommunity(**kwargs))
                    self.last_id += 1

            except:
                print("Issue with initializing network. Kwargs:")
                print(kwargs)

        elif (net_type == 'strict_com'):
            try:
                # Create weights and pass to network.
                self.mweights = make_master_weights(self.com_side * self.coms_per_side,
                    self.locality)
                kwargs.update({'mweights':self.mweights, 'locality':self.locality})

                for i in range(self.popsize):
                    kwargs.update({'ID':self.last_id})
                    self.population.append(StrictCommunity(**kwargs))
                    self.last_id += 1

            except:
                print("Issue with initializing network. Kwargs:")
                print(kwargs)

    def initialize(self):
        """
        Fills empty networks in population with initial connections.

        Parameters:
            None

        Returns:
            None.
        """

        # For each individual
        for i in self.population:

            # Initialize / populate them.
            i.initialize()

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
            i.mutate(node = None)

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
            
    def mu_modularity(self, network_num):
        within = 0
        for c in self.population[network_num].communities:
            idx = 0
            s_comm = set(c)
            for i in c:
                n_edges = self.population[network_num].adjL[c[idx]]
                comm_i = s_comm.intersection(n_edges)
                within += len(comm_i)
                idx += 1
        num_edges = sum(map(len, self.population[network_num].adjL.values()))
        outside = num_edges-within
        return (outside/num_edges)

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

