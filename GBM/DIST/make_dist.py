import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import elfi
import os, sys
import time
import pickle
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import GBM
import glob, os
from progress.bar import Bar
import multiprocessing as mp
from time import sleep

param = np.genfromtxt("../GRID/Examples.txt")

def ABMs(i):

    epsilon = param[i,1]
    lambdaC = param[i,0]
    Pdivi = param[i,3]
    Pdie = param[i,2]

    name = "E_"+str(epsilon)+"_L_"+str(lambdaC)+"_D_"+str(Pdivi)+"_cd_"+str(Pdie)

    if os.path.exists("./"+name):
        print("Directory exists")
    else:
        os.mkdir("./"+name)

    for j in range(250):

        store = "run_"+str(j)

        if os.path.exists("./"+name+"/"+store):
            print(store+" exists")
        else:
            # Create directory
            os.mkdir("./"+name+"/"+store)

        if os.path.exists("./"+name+"/"+store+"/Data_00099.h5"):
            print("No broken run")
        else:
            print("Running "+"./"+name+"/"+store)
            # Run simulations
            GBM.run_ABM("None", name+"/"+store, 201, 201, 100, 99, 5., lambdaC, 0.1, 5, 200, 8, 100, epsilon, Pdie, 0.02, Pdivi,  1., 1., False)

            print("sleeping to cool down")
            sleep(60.0)
            print("Now let's stop sleeping")

    return

sims = len(param)
NUM_CORES = 12

print("Simulations to run 250 times each: ", sims)

pool = mp.Pool(NUM_CORES)

results = pool.map(ABMs, [i for i in range(sims)])

pool.close()
