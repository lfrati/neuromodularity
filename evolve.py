"""
This file performs a very simple hillclimb evolution of growing networks.
"""

import copy
import time
import pickle
import numpy as np 
import pandas as pd 

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

	while (tstep > 0):

		# Create mutant population by copying and mutating.
		children = copy.deepcopy(parents)
		children.mutate()
		children.evaluate()

		# Replace parent with children if superior:
		for i in range(len(parents.population)):

			if (parents.population[i].fitness < children.population[i].fitness):
				parents.population[i] = children.population[i]

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
		print("\n----------- TIMESTEP", tstep, "-----------\n")

		print("Average fitness:", np.average([i.fitness 
			for i in parents.population]))

		print("Average mutations:", np.average([i.muts 
			for i in parents.population]))

		# Decrement
		tstep -= 1

	# Save database
	if (save):
		timestr = time.strftime("%Y-%m-%d_%H%M%S")
		with open('run' + timestr + ".pkl", 'wb') as fout:
			pickle.dump(db, fout)

opts = {"distr":"gauss", "locality":0.2} 
parents = Population(popsize=10, indsize=30, opts=opts, threshold=100)
parents.initialize(av_k=1, sample_type="local")

evolve(parents, 10, True)