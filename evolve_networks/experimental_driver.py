from experimental_tools import *

pop_params = {

	'small_gauss': 
	{'ID':0,
    'com_side':3, 
    'coms_per_side':3, 
    'threshold':4, 
    'fireweight':1, 
    'stype':'gaussian', 
    'popsize': 10, 
    'net_type':'gaussian_com', 
    'locality':0.25},

	'small_strict': 
	{'ID':0,
    'com_side':3, 
    'coms_per_side':3, 
    'threshold':4, 
    'fireweight':1, 
    'stype':'gaussian', 
    'popsize': 10, 
    'net_type':'strict_com', 
    'locality':0.25},
    # define more here.
}

stats = experiment(pop_params, 'genetic_tournament', 5000)
# stats = experiment(pop_params, 'hillclimb', 5000)

df = stats_to_df(stats)

# Make sure thee don't get overwritten!
get_adj_gen('small_gauss', 5000, df)
get_adj_gen('small_strict', 5000, df)