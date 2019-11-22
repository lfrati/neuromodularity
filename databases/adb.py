"""
This file is designed to find the aggregate performances across different
mu (network density) for different experimental tests.
"""

import pickle
import numpy as np 
import pandas as pd 

from reservoirlib.task import NbitRecallTask
from reservoirlib.distribution import Distribution
from reservoirlib.esn import DiscreteEchoStateNetwork
from reservoirlib.task import BinaryMemoryCapacityTask
from reservoirlib.experiment import BenchmarkExperiment
from reservoirlib.trainer import LeastSquaredErrorTrainer
from graphgen.lfr_generators import unweighted_directed_lfr_as_adj
from reservoirlib.generator import generate_reservoir_input_weights
from reservoirlib.generator import generate_adj_reservoir_from_adj_matrix

def binary_memory_mu(n = 100, seed_mu = 1, seed_dist = 1,  repeats = 3):
	"""
	This function finds the performance of a given graph on a binary memory
	task given at different values of mu (network density).

	Parameters:
		:int n: 		The number of nodes in the network.
		:int seed_mu: 	The seed to be passed to network params.
		:int repeats: 	Number of times to repeat mu gen. to get average.

	Returns:
		tbd
	"""

	# Declare parameters to be iterated
	mus = np.arange(.1, .5, .001)
	durations = [100,500,1000,1500,3000]

	# Initialize some consistently used vals.
	trainer = LeastSquaredErrorTrainer()
	dist = Distribution("uniform", {'low': -1.0, 'high': 1.0}, seed=seed_dist)

	# Initialize database
	bdb = pd.DataFrame(columns=[str(x) for x in durations])

	# For each potential mu
	for mu in mus:
		print("Evluating for mu", mu)

		# Define relevant params:
		net_pars = {'num_nodes': n,
                'average_k': 4,
                'max_degree': 4,
                'mu': mu,
                'com_size_min': 10,
                'com_size_max': 10,
                'seed': seed_mu,
                'transpose': True,
                'dtype': np.float32}

        # To be added to df:
		mustrip = {}

		# For each iteration:
		for d in durations:
			print("Evaluating for duration", d)

			# For each potential duration..
			avg = []
			for r in range(repeats):

				# Create a new graph and members
				graph, members = unweighted_directed_lfr_as_adj(**net_pars)
				reservoir = generate_adj_reservoir_from_adj_matrix(graph, dist)

				# Define new task:
				task = BinaryMemoryCapacityTask(duration=d, cut=2, num_lags=2, 
					shift=2)

				# Generate input weights for res.
				input_weights = generate_reservoir_input_weights(
					task.input_dimensions, reservoir_size=n, 
					input_fraction=0.4,distribution=dist, 
					by_dimension=True)

				# Generate discrete echo network
				esn = DiscreteEchoStateNetwork(reservoir, 
					input_weights=input_weights, 
					initial_state=dist, 
					neuron_type='sigmoid',
					neuron_pars = {'a':1, 'b':1, 'c':1, 'd':0, 'e':10}, 
					output_type='heaviside', 
					output_neuron_pars={'shape': (task.output_dimensions, 1), 
					'threshold': 0.0, 'newval': 1.0})

				# Generate experiment
				test_exp = BenchmarkExperiment(esn, task, trainer, num_training_trials=10, invert_target_of_training=False)

				# Train
				test_exp.train_model()

				# Append to average; only save memory capacity
				avg.append(test_exp.evaluate_model()[0])

			# Add to mustrip.
			mustrip[str(d)] = np.average(avg)

		# Add mustrip to df
		bdb = bdb.append(mustrip, ignore_index=True)

	return bdb

def bit_recall_mu(n = 100, seed_mu = 1, seed_dist = 1,  repeats = 3):
	"""
	This function finds the performance of a given graph on an n bit recall
	task given at different values of mu (network density).

	Parameters:
		:int n: 		The number of nodes in the network.
		:int seed_mu: 	The seed to be passed to network params.
		:int repeats: 	Number of times to repeat mu gen. to get average.

	Returns:
		tbd
	"""

	# Declare parameters to be iterated
	mus = np.arange(0, 1, .001)
	plengths = [1,2,3,4,5,6,7,8,9,10]

	# Initialize some consistently used vals.
	trainer = LeastSquaredErrorTrainer()
	dist = Distribution("uniform", {'low': -1.0, 'high': 1.0}, seed=seed_dist)

	# Initialize database
	bdb = pd.DataFrame(columns=[str(x) for x in plengths])

	# For each potential mu
	for mu in mus:
		print("Evaluating for mu", mu)

		# Define relevant params:
		net_pars = {'num_nodes': n,
                'average_k': 4,
                'max_degree': 4,
                'mu': mu,
                'com_size_min': 10,
                'com_size_max': 10,
                'seed': seed_mu,
                'transpose': True,
                'dtype': np.float32}

        # To be added to df:
		mustrip = {}

		# For each iteration:
		for p in plengths:
			print("Evaluating for plength", p)

			# For each potential duration..
			avg = []
			for r in range(repeats):

				# Create a new graph and members
				graph, members = unweighted_directed_lfr_as_adj(**net_pars)
				reservoir = generate_adj_reservoir_from_adj_matrix(graph, dist)

				# Define new task:
				task = NbitRecallTask(pattern_length=p, pattern_dimension=2, start_time=10, distraction_duration=6, cue_value=5, distractor_value=3, pattern_value=5, num_patterns=5, 
					seed=100)

				# Generate input weights for res.
				input_weights = generate_reservoir_input_weights(
					task.input_dimensions, reservoir_size=n, 
					input_fraction=0.4,distribution=dist, 
					by_dimension=True)

				# Generate discrete echo network
				esn = DiscreteEchoStateNetwork(reservoir, 
					input_weights=input_weights, 
					initial_state=dist, 
					neuron_type='sigmoid',
					neuron_pars = {'a':1, 'b':1, 'c':1, 'd':0, 'e':10},
					output_type='heaviside', 
					output_neuron_pars={'shape': (task.output_dimensions, 1), 
					'threshold': 0.0, 'newval': 1.0})

				# Generate experiment
				test_exp = BenchmarkExperiment(esn, task, trainer, num_training_trials=10, invert_target_of_training=False)

				# Train
				test_exp.train_model()

				# Append to average; only save memory capacity
				avg.append(test_exp.evaluate_model())

			# Add to mustrip.
			mustrip[str(p)] = np.average(avg)

		# Add mustrip to df
		bdb = bdb.append(mustrip, ignore_index=True)

	return bdb