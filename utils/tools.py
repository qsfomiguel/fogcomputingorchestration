import numpy as np
import pandas as pd
import os
import csv

def set_tools_seed(seed=None):
	global np_tools_random # change the np_tools_random seed
	np_tools_random = np.random.RandomState(seed=seed)

def uniform_rand_float(m=1):
	return np_tools_random.rand()*m

def uniform_rand_int(low=0, high=1):
    return np_tools_random.randint(low=low, high=high)

def uniform_rand_array(size=1):
	return np_tools_random.random(size)

def uniform_rand_choice(l):
	return np_tools_random.choice(l)

def random_seed_primes(max_val=100):
	return [num for num in range(2, max_val) if all(num%i!=0 for i in range(2,int(np.sqrt(num))+1)) ]
