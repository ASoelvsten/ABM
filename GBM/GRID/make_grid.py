import sobol_seq
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import GBM
import multiprocessing as mp

def grid_select(param_min, param_max, n):

    ndim = len(param_min)

    interval = param_max-param_min

    sobols = param_min + interval*sobol_seq.i4_sobol_generate(ndim, n, 1)

    return sobols, ndim

def simulations(newdir, params):

    os.mkdir(newdir)
    lambdaC = params[0]
    epsilon = params[1]
    Pdie = params[2]
    Pdivi = params[3]

    GBM.run_ABM("None", newdir[2:], 201, 201, 100, 99, 5., lambdaC, 0.1, 5, 200, 8, 100, epsilon, Pdie, 0.02, Pdivi,  1., 1., False)

params = ["lam","eps","die","divi"]
param_min = [0.01,0.01,0.01,0.01]
param_max = [0.50,0.50,0.50,0.50]
#param_min = [0.05,0.05,0.05,0.05]
#param_max = [0.45,0.45,0.45,0.45]

sims = 10
NUM_CORES = 10

grid, ndim = grid_select(np.asarray(param_min), np.asarray(param_max), sims)

print(grid)

def run_grid(i):

    newdir = "./LIB2/GBM"
    for j in range(ndim):
        newdir += "_" + params[j] + str(grid[i][j])

    if not os.path.isdir(newdir):
        simulations(newdir, grid[i])
        ff = open(newdir+"/varied.txt", "w")
        for j in range(ndim):
            ff.write(params[j]+": " + str(grid[i][j]) + "\n")
        ff.close()
    else:
        print("Pre-existing simulations, skipped", newdir)

pool = mp.Pool(NUM_CORES)

results = pool.map(run_grid, [i for i in range(sims)])

pool.close()
