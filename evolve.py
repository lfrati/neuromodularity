"""
This file performs a very simple hillclimb evolution of growing networks.
"""

import copy
import time
import pickle
import numpy as np 
import pandas as pd 

#for calculating modularity
import networkx as nx
from modularity_maximization import partition #you'll have to pip install these from https://zhiyzuo.github.io/python-modularity-maximization/
from modularity_maximization.utils import get_modularity


from Population import Population

def evolve(parents, tstep, save):
        """
        Perform an evolutionary run on an initial population.

        Parameters:
            :population parents: 		Initial population of parent networks.
            :int tsteps: 				Generations to evolve through.
            :bool save: 				Saves a dataframe of run if True.

        Returns:
            None
        """

        # Create database
        db = pd.DataFrame(columns = ["N", "Opts", "Adjlist", "Threshold"])

        # Find initial fitnesses,
        parents.evaluate()
        
        # Find modularity of initial parent
        adj_for_mod = parents.population[0].adjL
        G=nx.DiGraph(adj_for_mod)
        comm_dict = partition(G)
        print(get_modularity(G, comm_dict))
        
        
        
        while (tstep > 0):

            # Create mutant population by copying and mutating.
            children = copy.deepcopy(parents)
            children.mutate()
            children.evaluate()

            # Replace parent with children if superior:
            for i in range(len(parents.population)):

                if (parents.population[i].fitness <= children.population[i].fitness):
                    parents.population[i] = children.population[i]
                    print("replaced parent")

                # Increase age by 1
                parents.population[i].age += 1

            # Once everything has been replaced, sort so that best is in front.
                # Note, this does not really matter within the context of this alg.
                # I'm just reusing code.

            parents.population.sort(key=lambda x: x.fitness, reverse=True)

            # Save best performing at that timestep.
            db = db.append({
                "N":parents.population[0].N,
                "Opts":parents.population[0].opts,
                "Adjlist":parents.population[0].adjL,
                "Threhold":parents.population[0].threshold
                }, ignore_index=True)

            # Print some stats:
            avg_fit = np.average([i.fitness 
                for i in parents.population])

            avg_mut = np.average([i.muts 
                for i in parents.population])

            print("--- Timestep:", tstep, "|", "Avg Fit:", 
                avg_fit, "|", "Avg Mut:", avg_mut, "---", end='\r')

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

opts = {"distr":"gauss", "locality":0.3} 
parents = Population(popsize=1, indsize=9, opts=opts, threshold=140)
tot_num_edge = (9*(9-1))*9 #when indsize is 9 and there are 9 clusters
parents.initialize() #start locality network but growth is uniform

num_edges1 = sum(map(len, parents.population[0].adjL.values()))

parents.population[0].show_grid()
evolve(parents, 5000, False)
parents.population[0].show_grid()

num_edges2 = sum(map(len, parents.population[0].adjL.values()))

# Clear print
print("\n")
