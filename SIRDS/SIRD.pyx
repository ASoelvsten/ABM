#cython: boundscheck=False
#cython: language_level=3
#cython: wraparound=False

import scipy.spatial.distance as d
from PIL import Image, ImageDraw
import glob
import cython
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import matplotlib.pyplot as plt
import random
import time
import h5py
import os
from more_itertools import flatten
import random
from scipy.stats import expon

cimport cython
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport exp as c_exp

def SIRABM(str store, double dt, int hotspots, double skip, double Population, double Pdie, double Pinf, double REN, double tinf, double immune, double t_max, bint plotting):

    cdef str cwd = os.getcwd()
    cdef extern from "stdlib.h":
        double drand48()

    cdef int gsize, idummy1, idummy2, idummy3, idummy4, i, j, k
    cdef double t = 0
    cdef double dummy, endtime, Pdummy, modu, dummy2, treco, tsus
    cdef list neigh, NN, Sick, encounters
    cdef np.ndarray x, y, X, Y, points, SUS, INDEX, INF, DEAD, RECO, p, mask, ENCO, data
    cdef np.ndarray TRECO, TSUS
    cdef str logfile 
    cdef double tstart = time.time()

    x = np.linspace(0,round(np.sqrt(Population))-1,round(np.sqrt(Population)))
    y = x.copy()

    Y, X = np.meshgrid(y,x)
    points = np.c_[X.ravel(), Y.ravel()]

    NN = []

    if skip > 0:
        for i in range(int(skip*Population)):
            points = np.delete(points, random.randint(0, int(len(points)-1)), axis=0)

    tree = cKDTree(points)
    gsize = len(points)

    SUS = np.ones(gsize)
    INDEX = np.arange(gsize)
    INF = np.zeros(gsize)
    Sick = random.sample(range(1, gsize), hotspots)
    INF[Sick] = 1e-8
    SUS[Sick] = 0
    DEAD = np.zeros(gsize)
    RECO = np.zeros(gsize)
    TRECO = expon.rvs(scale=tinf, size=gsize, random_state=None)
    TSUS = expon.rvs(scale=immune, size=gsize, random_state=None)

    print("Population size: %i" % (len(SUS)))

    logfile = cwd + "/" + store + "/history.log"
    if not os.path.isfile(logfile):
        ff = open(logfile, "w")
        ff.write("# Summary file for ABM. \n")
        ff.write("# Timestep, infected, susceptible, recovered (immune), dead \n")
        ff.close()

    for p in points:
        neigh = tree.query_ball_point(p, 1.5)
        NN.extend([neigh])

    NoNe = []

    for nn in NN:
        NoNe.extend([len(nn)])

    print("Mean no. of neighbours:", np.mean(NoNe))

    print("Finished initialization after %.2f seconds" % (time.time()-tstart))

    idummy1 = len(INF[INF>0])
    idummy2 = len(SUS[SUS>0])
    idummy3 = len(RECO[RECO>0])
    idummy4 = len(DEAD[DEAD>0])

    ff = open(logfile, "a")
    ff.write("%.2f %i %i %i %i \n" % (t, idummy1, idummy2, idummy3, idummy4))
    ff.close()

    while (len(INDEX[INF>0])>0 and t < t_max):
        t = t+dt
        mask = INF > 0.0
        idummy2 = len(INDEX[mask])
        for ii in range(idummy2):
            i = INDEX[mask][ii]
            # Meeting with common crowd
            neigh = NN[i]
            idummy4 = len(neigh)
            for jj in range(idummy4):
                j = neigh[jj]
                dummy = drand48()
                Pdummy = float(Pinf*dt*SUS[j])
                if dummy < Pdummy and j!=i:
                    INF[j]=dt
                    SUS[j]=0.0

            # Meeting random people
            modu = float(np.mod(np.round(t,decimals=1),1))
            if modu == 0:
                idummy1 = int(np.floor(REN))
                dummy = drand48()
                Penc = float(REN-float(idummy1))
                if dummy < Penc:
                    idummy1 += 1
                if idummy1 >0:
                    ENCO = INDEX[~np.isin(INDEX,neigh)]
                    idummy3 = int(len(ENCO))
                    encounters = list(ENCO[random.sample(range(0, idummy3), idummy1)])
                    idummy3 = len(encounters)
                    for jj in range(idummy3):
                        j = encounters[jj]
                        dummy = drand48()
                        Pdummy = float(Pinf*dt*SUS[j])
                        if dummy < Pdummy:
                            INF[j]=dt
                            SUS[j]=0.0

            treco = TRECO[i]
            dummy2 = float(INF[i])
            if dummy2>=treco:
                dummy = drand48()
                if dummy < Pdie:
                    DEAD[i] = 1.0
                    INF[i] = 0.0
                else:
                    INF[i]=0.0
                    RECO[i]=1e-8
            else:
                INF[i] += dt

        mask = RECO > 0.0
        idummy2 = len(INDEX[mask])
        for ii in range(idummy2):
            i = INDEX[mask][ii]
            dummy2 = RECO[i]
            tsus = TSUS[i]
            if dummy2>=tsus:
                RECO[i]=0.0
                SUS[i]=1.0
            else:
                RECO[i] +=dt

        idummy1 = len(INF[INF>0])
        idummy2 = len(SUS[SUS>0])
        idummy3 = len(RECO[RECO>0])
        idummy4 = len(DEAD[DEAD>0])
        endtime = time.time()-tstart
        print(np.round(endtime,decimals=1),np.round(t,decimals=2),idummy1,idummy2,idummy3,idummy4)

        i = int(t/dt)

        exten = "%05d" % i

        if plotting:
            plt.figure()
            for k in INDEX[INF>0.0]:
               plt.plot(points[k][0],points[k][1],"r.")
            for k in INDEX[SUS>0.0]:
                plt.plot(points[k][0],points[k][1],"c.")
            for k in INDEX[RECO>0.0]:
                plt.plot(points[k][0],points[k][1],"g.")
            for k in INDEX[DEAD>0.0]:
                plt.plot(points[k][0],points[k][1],"k.")
            plt.xlabel(r'$\mathrm{x\, lattice \, position}$',fontsize=16)
            plt.ylabel(r"$\mathrm{y\, lattice \, position}$",fontsize=16)
            plt.xticks(size=16)
            plt.yticks(size=16)
            plt.gcf().subplots_adjust(bottom=0.18)
            plt.gcf().subplots_adjust(left=0.18)
            plt.savefig(cwd+"/"+store+"/Snapshot_"+exten+".png",bbox_inches='tight')
            plt.close()

        ff = open(logfile, "a")
        ff.write("%.2f %i %i %i %i \n" % (t, idummy1, idummy2, idummy3, idummy4))
        ff.close()

    data = np.genfromtxt(logfile)
    plt.plot(data[:,0],data[:,2],'c',label="Susceptible")
    plt.plot(data[:,0],data[:,1],'r--',label="Infected")
    plt.plot(data[:,0],data[:,3],'g-.',label="Recovered")
    plt.plot(data[:,0],data[:,4],'k',label="Deceased")
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.legend(loc="best",fontsize=16)
    plt.xlabel(r"$\mathrm{Time\,\, [days]}$",fontsize=16)
    plt.ylabel(r"$\mathrm{Population}$",fontsize=16)
    plt.gcf().subplots_adjust(bottom=0.18)
    plt.gcf().subplots_adjust(left=0.18)
    plt.savefig(cwd + "/" + store + "/run.png",bbox_inches='tight')
