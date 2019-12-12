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

### EVOLUTIONARY FUNCTIONS

def hillclimb(parents, tstep):
    """
    Perform an evolutionary run on an initial population using a hill-climb
    algorithm.

    Each parent is compared to its child. If the child is superior, it
    replaces the parent.

    Parameters:
        :population parents:        Initial population of parent networks.
        :int tsteps:                Generations to evolve through.

    Returns:
        None
    """

    stats = {x:[] for x in ['avg_fit', 'avg_mod', 'best_fit', 'best_mod', 'best_net_adj',
    'best_net_astates']}

    # Find initial fitnesses,
    parents.evaluate()
    parents.find_modularity()

    
    while (tstep > 0):

        # Create mutant population by copying and mutating.
        children = copy.deepcopy(parents)
        children.mutate()
        children.evaluate()
        children.find_modularity()

        # Replace parent with children if superior:
        for i in range(len(parents.population)):

            #also replaces parent when fitness is the same
            if (parents.population[i].fitness <= 
                children.population[i].fitness): 
                parents.population[i] = children.population[i]

            # Increase age by 1
            parents.population[i].age += 1

        parents.population.sort(key=lambda x: x.fitness)

        avg_fit = np.average([i.fitness for i in parents.population])
        avg_mod = np.average([i.modularity for i in parents.population])

        stats['avg_fit'].append(avg_fit)
        stats['avg_mod'].append(avg_mod)

        stats['best_fit'].append(parents.population[-1].fitness)
        stats['best_mod'].append(parents.population[-1].modularity)

        stats['best_adj'].append(parents.population[-1].adjL)
        stats['best_net_astates'].append(parents.population[-1].astates)

        print("--- tstep:", tstep, "|", "avg fit", round(avg_fit, 4), "|",
        "avg modu:", round(avg_mod,4), "---", end='\r')

        # Decrement
        tstep -= 1

    return stats

def genetic(parents, tstep, style):
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
        :str style:                 Style of genetic evolution.

    Returns:
        None
    """

    stats = {x:[] for x in ['avg_fit', 'avg_mod', 'best_fit', 'best_mod', 'best_net_adj',
        'best_net_astates']}
        
    # Generate rank weights (+1 because range is used)
    if (style == "ranked"):
        ranks_a = ranked(2, len(parents.population) + 1)

        while (tstep > 0):
            parents.evaluate()
            parents.find_modularity()
            parents.population.sort(key=lambda x: x.fitness)

            ### Print some stats:
            avg_fit = np.average([i.fitness for i in parents.population])
            avg_mod = np.average([i.modularity for i in parents.population])

            stats['avg_fit'].append(avg_fit)
            stats['avg_mod'].append(avg_mod)

            stats['best_fit'].append(parents.population[-1].fitness)
            stats['best_mod'].append(parents.population[-1].modularity)

            stats['best_net_adj'].append(parents.population[-1].adjL)
            stats['best_net_astates'].append(parents.population[-1].astates)

            print("--- tstep:", tstep, "|", "avg fit", round(avg_fit, 4), "|",
            "avg modu:", round(avg_mod,4), "---", end='\r')

            # Select offspring
            offspring = list(np.random.choice(
                a = parents.population,
                p = ranks_a,
                replace = True,
                size = len(parents.population)))

            # Replace with copies to prevent redundant mutation:
            offspring = [copy.deepcopy(x) for x in offspring]

            # Mutate offspring
            for i in offspring:
                i.mutate() 
                i.evaluate()

            # Find best of mutated original individuals
            pool = offspring + parents.population 
            pool.sort(key=lambda x: x.fitness, reverse = True)

            # Select best:
            parents.population = pool[0: len(parents.population)]

            tstep -= 1

    elif (style == "tournament"):
        while (tstep > 0):
            parents.evaluate()
            parents.find_modularity()

            ### Print some stats:
            avg_fit = np.average([i.fitness for i in parents.population])
            avg_mod = np.average([i.modularity for i in parents.population])

            stats['avg_fit'].append(avg_fit)
            stats['avg_mod'].append(avg_mod)

            stats['best_fit'].append(parents.population[-1].fitness)
            stats['best_mod'].append(parents.population[-1].modularity)

            stats['best_net_adj'].append(parents.population[-1].adjL)
            stats['best_net_astates'].append(parents.population[-1].astates)

            print("--- tstep:", tstep, "|", "avg fit", round(avg_fit, 4), "|",
            "avg modu:", round(avg_mod,4), "---", end='\r')


            parents.population.sort(key=lambda x: x.fitness)
            children = copy.deepcopy(parents)

            # Preserve best of this generation: elitism
            offspring = [parents.population[-1]]
            for i in range(len(parents.population) - 1):

                # Tournament selection: Take best of 5 (tunable)
                sample = list(np.random.choice(a = parents.population, replace = True, size = 5))
                sample.sort(key= lambda x: x.fitness)

                # Find best of tournament and mutate.
                best = copy.deepcopy(sample[-1])
                best.mutate()
                offspring.append(best)

            for i in offspring:
                i.evaluate()

            children.population = offspring 
            parents = children

            tstep -= 1

    return stats 

### Driver and examples

kwargs = {'ID':0,
        'com_side':3, 
        'coms_per_side':3, 
        'threshold':4, 
        'fireweight':1, 
        'stype':'gaussian', 
        'popsize': 5, 
        'net_type':'gaussian_com', 
        'locality':0.25}

# parents = Population(**kwargs)
# parents.initialize()
# stats = hillclimb(parents, 1000, False)