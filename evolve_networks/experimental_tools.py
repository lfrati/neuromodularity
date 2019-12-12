"""
Essentially a drive file for experimental evolution,
allowing multiple experiments to be run in parallel.

Pass a dict with keys = labels, values = kwargs to the experiment
and it will create corresponding populations, run them for an evolutionary
experiment, and save results.
"""

import csv
import pickle
import joblib
import pandas as pd
from evolve import *

def create_pops(pop_params):
	"""
	Creates populations relevant kwargs dictionaries.

	Parameters:
		:dict pop_params: 	dict of kwargs passed to networks.

	Returns:
		:list populations: 	List of populations.
	"""

	pops = []
	for i in list(pop_params.values()):
		pops.append(Population(**i))

	return pops

def experiment(pop_params, algorithm, timesteps):
	"""
	Runs experiments in parallel for each potential population described
	by parameters.

	Parameters:
		:list pop_params: 	List of kwargs passed to networks.
		:str algorithm: 	The genetic algorithm to be used.
		:int timesteps: 	How long to run the algorithms.

	Returns:
		:dict results: 			Dict of results.
	"""

	pops = create_pops(pop_params)
	for i in pops:
		i.initialize()

	if (algorithm == 'hillclimb'):
		results = joblib.Parallel(n_jobs=-1)(
			joblib.delayed(hillclimb)(
				tstep = timesteps, parents = x) for x in pops)

	elif (algorithm == 'genetic_ranked'):
		results = joblib.Parallel(n_jobs=-1)(
			joblib.delayed(genetic)(
				tstep = timesteps, style = 'ranked', parents = x) for x in pops)

	elif (algorithm == 'genetic_tournament'):
		results = joblib.Parallel(n_jobs=-1)(
			joblib.delayed(genetic)(
				tstep = timesteps, style = 'tournament', parents = x) for x in pops)

	stats = {x:y for x,y in zip(list(pop_params.keys()), results)}

	return stats

def stats_to_df(stats):
	"""
	Turns the returned stats into a dataframe, which can be saved or loaded as a pickled file.
	This is much more flexible (and legible) than a CSV.

	Represents values over the entirety of an evolutionary run.
	"""

	index = list(stats.keys())
	columns = list(stats[index[0]].keys())

	df = pd.DataFrame(index = index, columns = columns)

	for i in index:
		for c in columns:
			df.loc[i, c] = stats[i][c]

	return df

def get_adj_gen(netkey, gen, df):
	"""
	Returns an adjacency list of a specific generation of 
	the evolutionary run from the best individual in the
	population.

	Parameters:
		:str netkey: 		The network key.
		:int gen: 			The generation of focus.
		:dataframe df: 		The pandas dataframe.
	"""

	adjLs = df.loc[netkey, 'best_net_adj']
	adjL = adjLs[gen]

	# Exports gephi-friendly format of CSV.
	fout = open(netkey + '_' + str(gen) + '.csv', 'w')
	writer = csv.writer(fout)
	for i in adjL.keys():
		row = [i] + list(adjL[i])
		writer.writerow(row)

	fout.close()



### Example use:
'''
pop_params = {

	'small_gauss': 
	{'ID':0,
    'com_side':3, 
    'coms_per_side':3, 
    'threshold':4, 
    'fireweight':1, 
    'stype':'gaussian', 
    'popsize': 5, 
    'net_type':'gaussian_com', 
    'locality':0.25},

	'small_strict': 
	{'ID':0,
    'com_side':3, 
    'coms_per_side':3, 
    'threshold':4, 
    'fireweight':1, 
    'stype':'gaussian', 
    'popsize': 46, 
    'net_type':'strict_com', 
    'locality':0.25},

    'big_gaussian': 
    {'ID':0,
    'com_side':3, 
    'coms_per_side':3, 
    'threshold':100, 
    'fireweight':50, 
    'stype':'gaussian', 
    'popsize': 400, 
    'net_type':'gaussian_com', 
    'locality':0.5}

    # define more here.
}

stats = experiment(pop_params, "genetic", 1000)

stats['big_gaussian']['avg_fit'] # Average fitness for every generation of big_gaussian
stats['small_strict'].keys() 	 # Shows available stats for small_strict

plt.plot(stats['big_gaussian']['avg_fit'], np.arange(1000))

# Load data as a dataframe, where it can be a bit more understandable:

df = stats_to_df(stats)

# Index specific row, column of data frame:

df.loc[row, column]

# Show entirety of column

df[column]

# Sort by attribute

df.loc[df[column] == attribute]

# Export run to csv

df.to_csv('name.txt')

# Pickle dataframe

df.to_pickle('name.pkl')

# Unpickle dataframe

df = pd.read_pickle('name.pkl')

# Find a specific adjlist of tested network (network) for a generation
# exports as netkey_gen.csv

get_adj_gen('small_gaussian', 50, df)

# this would create 'small_gaussian_50.csv'

'''
