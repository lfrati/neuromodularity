"""
This file performs a very simple hillclimb evolution of growing networks.
"""

import copy
import time
import random
import pickle
import numpy as np 
import pandas as pd 

# for calculating modularity
import networkx as nx

# you'll have to pip install these from 
# https://zhiyzuo.github.io/python-modularity-maximization/


from Population import Population

### HELPER FUNCTIONS

def ranked(s, mu): 
    """
    Generates a fitness proportionate selection matrix.
    
    Parameters:
        :int s:         Number of expected children from 
        :int mu:        The population size
    """

    ranks = [] 
    for i in range(1, mu): 
        term1 = (2 - s) / mu  
        term2a = (2 * i * (2-1)) 
        term2b = (mu * ( mu - 1)) 
        term2 = term2a / term2b  
        ranks.append(term1 + term2) 
    return ranks

def hillclimb(parents, tstep, save):
        """
        Perform an evolutionary run on an initial population using a hill-climb
        algorithm.

        Each parent is compared to its child. If the child is superior, it
        replaces the parent.

        Parameters:
            :population parents: 		Initial population of parent networks.
            :int tsteps: 				Generations to evolve through.
            :bool save: 				Saves a dataframe of run if True.

        Returns:
            None
        """

        # Find initial fitnesses,
        parents.evaluate()
        
        while (tstep > 0):

            # Create mutant population by copying and mutating.
            children = copy.deepcopy(parents)
            children.mutate()
            children.evaluate()

            # Replace parent with children if superior:
            for i in range(len(parents.population)):

                #also replaces parent when fitness is the same
                if (parents.population[i].fitness <= 
                    children.population[i].fitness): 
                    parents.population[i] = children.population[i]

                # Increase age by 1
                parents.population[i].age += 1

            parents.population.sort(key=lambda x: x.fitness, reverse=True)

            # Print some stats:
            avg_fit = np.average([i.fitness 
                for i in parents.population])

            print("--- Timestep:", tstep, "|", "Avg Fit:", 
                round(avg_fit, 8), "|", "---", end='\r')

            # Decrement
            tstep -= 1

def genetic(parents, tstep, save):
    """
    Performs evolution on a population of robots using a genetic algorithm.

    Essentially, the number of offspring a parent has is proportional to that
    parent's fitness. Higher fitness = higher number of children.

    Since fitness can be arbitrarily low or high, fitness-proportionate 
    selection is based on rank instead of fitness alone.

    Essentially:

    Find fitnesses
    Rank in order of fitness
    While child population is not full:
        Select individual from parent population, given fitness-prop probs
    Mutate whole child population, replace parent population with children.

    Parameters:
        :population parents:        The parent population being evolved.
        :int tstep:                 The number of timesteps to evolve for.
        :bool save:                 Whether or not a sum of the run is saved.

    Returns:
        None
    """

    # Generate rank weights (+1 because range is used)
    ranks = ranked(2, len(parents.population) + 1)

    mu_modularity = parents.mu_modularity(network_num = 0) #evaluates modularity of a representative inidivual (0)
    print(mu_modularity)
    
    while (tstep > 0):

        # Rank parents, find probs:
        parents.evaluate()

        # Print some stats:
        avg_fit = np.average([x.fitness for x in parents.population])

        print("--- Timestep:", tstep, "|", "Avg Fit:", 
                round(avg_fit, 8), "|", "---", end='\r')

        parents.population.sort(key=lambda x: x.fitness)
        kiddos = np.random.choice(
            a = parents.population, 
            p = ranks, 
            replace = True,
            size = len(parents.population))

        # Must be distinct memory places or else mutation will occur on same 
        # network multiple times.
        kiddos = [copy.deepcopy(x) for x in kiddos]

        parents.population = kiddos
        parents.mutate()

        # Decrement
        tstep -= 1
    
    idx = 0
    pop_mod = []
    for parent in parents.population:
        mu_modularity = parents.mu_modularity(network_num = idx) #evaluates modularity of inidivual 0
        pop_mod.append(mu_modularity)
        idx += 1
    print(np.mean(pop_mod))

### Driver and examples

kwargs = {'ID':0,
        'com_side':3, 
        'coms_per_side':3, 
        'threshold':140, 
        'fireweight':20, 
        'stype':'gaussian', 
        'popsize': 100, 
        'net_type':'gaussian_com', 
        'locality':0.25}

parents = Population(**kwargs)
parents.initialize()
genetic(parents, 10000, False)