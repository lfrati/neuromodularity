from Population import *

gkwargs = {'ID':0,
        'com_side':3, 
        'coms_per_side':3, 
        'threshold':100, 
        'fireweight':20, 
        'stype':'gaussian', 
        'popsize':10, 
        'net_type':'gaussian_com', 
        'locality':0.5}

skwargs = {'ID':0,
        'com_side':3, 
        'coms_per_side':3, 
        'threshold':100, 
        'fireweight':20, 
        'stype':'gaussian', 
        'popsize':10, 
        'net_type':'strict_com', 
        'locality':0.5}

s_pop = Population(**skwargs)
g_pop = Population(**gkwargs)

s_pop.initialize()
g_pop.initialize()

s_pop.population[0].show_grid(True, 10)
g_pop.population[0].show_grid(True, 10)