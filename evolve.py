"""
This file performs a very simple hillclimb evolution of growing networks.
"""

import copy
import time
import pickle
import numpy as np 
import pandas as pd 

# for calculating modularity
import networkx as nx

# you'll have to pip install these from 
# https://zhiyzuo.github.io/python-modularity-maximization/


from Population import Population

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

        # Create database
        db = pd.DataFrame(columns = ["N", "Dist", "Locality", 
            "Adjlist", "Threshold", "Comshape"])

        # Find initial fitnesses,
        parents.evaluate()
        
        mu_modularity = parents.mu_modularity(network_num = 0) #evaluates modularity of inidivual 0
        print(mu_modularity)
        
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

            # Save best performing at that timestep.
            # db = db.append({
            #     "N":parents.population[0].N,
            #     "Dist":parents.population[0].dist,
            #     "Locality":parents.population[0].locality,
            #     "Adjlist":parents.population[0].adjL,
            #     "Threhold":parents.population[0].threshold,
            #     "Comshape":parents.population[0].comshape
            #     }, ignore_index=True)

            # Print some stats:
            avg_fit = np.average([i.fitness 
                for i in parents.population])

            avg_mut = 0
            # avg_mut = np.average([i.muts 
            #     for i in parents.population])

            print("--- Timestep:", tstep, "|", "Avg Fit:", 
                avg_fit, "|", "Avg Mut:", avg_mut, "---", end='\r')

            # Decrement
            tstep -= 1
            
        mu_modularity = parents.mu_modularity(network_num = 0) #evaluates modularity of inidivual 0
        print(mu_modularity)

        # Save database
        if (save):
            timestr = time.strftime("%Y-%m-%d_%H%M%S")
            with open('run' + timestr + ".pkl", 'wb') as fout:
                pickle.dump(db, fout)

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

    # Create database
    db = pd.DataFrame(columns = ["N", "Dist", "Locality", 
        "Adjlist", "Threshold", "Comshape"])

    # Find initial fitnesses,
    parents.evaluate()
    
    # Find modularity of initial parent
    adj_for_mod = parents.population[0].adjL
    G=nx.DiGraph(adj_for_mod)
    comm_dict = partition(G)
    print(get_modularity(G, comm_dict))

    while (tstep > 0):

        # Rank parents, find probs:
        parents.population.sort(key=lambda x: x.fitness, reverse=True)

        # # Save best performing at that timestep.
        # db = db.append({
        #     "N":parents.population[0].N,
        #     "Dist":parents.population[0].dist,
        #     "Locality":parents.population[0].locality,
        #     "Adjlist":parents.population[0].adjL,
        #     "Threhold":parents.population[0].threshold,
        #     "Comshape":parents.population[0].comshape
        #     }, ignore_index=True)

        # Find rank-prop probabilities
        
        ### TODO: Find a way to easily implement rank based selection

        print(probs)
        print(sum(probs))

        # Create dummy population, replace its individuals with fitness prop
        children = copy.deepcopy(parents)
        survive = np.random.choice(children.population, 
            len(children.population), p = probs, replace = True)

        for i in range(len(children.population)):
            children.population[i] = survive[i]

        # Mutate them all
        children.mutate()

        # Replace parents with children
        parents = children

        # Print some stats:
        avg_fit = np.average([i.fitness 
            for i in parents.population])

        avg_mut = 0
        # avg_mut = np.average([i.muts 
        #     for i in parents.population])

        print("--- Timestep:", tstep, "|", "Avg Fit:", 
            avg_fit, "|", "Avg Mut:", avg_mut, "---")

        # Decrement
        tstep -= 1

    # Find modularity of last parent
    adj_for_mod = parents.population[0].adjL
    G=nx.DiGraph(adj_for_mod)
    comm_dict = partition(G)
    print(get_modularity(G, comm_dict))

    # Save database
    if (save):
        timestr = time.strftime("%Y-%m-%d_%H%M%S")
        with open('run' + timestr + ".pkl", 'wb') as fout:
            pickle.dump(db, fout)


### Driver and examples

kwargs = {'ID':0,
        'com_side':3, 
        'coms_per_side':3, 
        'threshold':140, 
        'fireweight':20, 
        'stype':'gaussian', 
        'popsize':10, 
        'net_type':'strict_com', 
        'locality':0.25}

parents = Population(**kwargs)
parents.initialize()

hillclimb(parents, 10000, False)
parents.population[0].show_grid(True, 10)