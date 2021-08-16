#cython: boundscheck=False
#cython: language_level=3
#cython: wraparound=False

import cython
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import matplotlib.pyplot as plt
import random
import time
from fipy import CellVariable, Grid2D, Viewer, TransientTerm, DiffusionTerm
import h5py
import os
from more_itertools import flatten

cimport cython
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport exp as c_exp

###############################################
#   ____       ____     __  __                # 
#  / ___|__ _ / ___|___|  \/  |               #
# | |   / _` | |   / _ \ |\/| |               #
# | |__| (_| | |__|  __/ |  | |               #
#  \____\__,_|\____\___|_|  |_|               #
#                                             #
###############################################

# Function to reduce grid resolution and create clusters of dead and quiescent cells

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def Dead_cluster(np.ndarray GSC, np.ndarray GPP, np.ndarray GDS, np.ndarray Alive, np.ndarray Cell_pop, np.ndarray Dead, np.ndarray Rim, list Xs_r, list Ys_r, list Discarded, np.ndarray points, tri, np.ndarray index, np.ndarray indptr, np.ndarray indices, int Dup, int Qup, double trapped):

    cdef list neighbours = []
    cdef list dead_neighbours = []
    cdef list qui_GSC = []
    cdef list qui_GPP = []
    cdef list qui_GDS = []
    cdef list cluster = []

    cdef np.ndarray neigh
    cdef int j, i
    cdef int dsize, nsize, nsize_alive, nsize_trap, nsize_ndead, csize
    cdef double deadcount, alivecount, cellcount

    dsize = len(Dead)

    for j in range(dsize):
        neigh = indices[indptr[j]:indptr[j+1]]
        nsize = len(neigh)
        deadcount = Dead[j]
        alivecount = Alive[j]
        nsize_ndead = len(neigh[Dead[neigh] == 0])
        nsize_alive = len(neigh[Alive[neigh] > 0 ])
        cellcount = Cell_pop[j]
        nsize_trap = len(neigh[Rim[neigh] <= trapped])

        # If there are only dead cells around a dead cell then you might make a cluster of at least Dup.
        if j not in neighbours and deadcount > 0. and deadcount < float(Dup) and alivecount == 0 and nsize > 0 and nsize_ndead == 0 and nsize_alive ==0:
            neigh = neigh[Dead[neigh] <= Dup]
            neigh = neigh[np.in1d(neigh,neighbours,invert=True)]
            nsize = len(neigh)
            if nsize > 0:
                neighbours.extend(list(neigh))
                dead_neighbours.extend([np.sum(Dead[neigh])])
                qui_GSC.extend([0])
                qui_GPP.extend([0])
                qui_GDS.extend([0])
                cluster.extend([j])

        # Otherwise, you might make a cluster of dead and living cells (GSC, GPP, GDS) of at least Qup cells.
        elif j not in neighbours and nsize > 0. and (cellcount > 0. and cellcount < float(Qup)) and nsize_trap == 0:
            neigh = neigh[Cell_pop[neigh] <= Qup]
            neigh = neigh[np.in1d(neigh,neighbours,invert=True)]
            nsize = len(neigh)
            if nsize > 0:
                neighbours.extend(list(neigh))
                dead_neighbours.extend([np.sum(Dead[neigh])])
                qui_GSC.extend([np.sum(GSC[neigh])])
                qui_GPP.extend([np.sum(GPP[neigh])])
                qui_GDS.extend([np.sum(GDS[neigh])])
                cluster.extend([j])

    csize = len(cluster)

    # Move cells to cluster position.
    for j in range(csize):
        i = cluster[j]
        Dead[i] += dead_neighbours[j]
        GSC[i] += qui_GSC[j]
        GPP[i] += qui_GPP[j]
        GDS[i] += qui_GDS[j]     

    # Delete previous positions.
    for j in sorted(set(neighbours), reverse=True):
        Dead = np.delete(Dead, j)
        Rim = np.delete(Rim, j)
        GSC = np.delete(GSC, j)
        GPP = np.delete(GPP, j)
        GDS = np.delete(GDS, j)
        del Xs_r[j]
        del Ys_r[j]
        Discarded.append(points[j])

    Alive = GSC + GPP + GDS
    Cell_pop = Dead + Alive
 
    # Re-initialize certain arrays. 
    points = np.append(Xs_r,Ys_r,axis=1)
    
    tri = Delaunay(points)
 
    (indptr, indices) = tri.vertex_neighbor_vertices

    gsize = len(points)
    index = np.asarray(range(gsize))

    return gsize, GSC, GPP, GDS, Alive, Cell_pop, Dead, Rim, Xs_r, Ys_r, Discarded, points, tri, index, indptr, indices

#=========================================

# Rates for stochastic events

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def stocev(np.ndarray Ox_irr, double O2_crit, double Pdie, double Pprol, double Pdivi):

    cdef Py_ssize_t gsize = len(Ox_irr)
    cdef np.ndarray Pmove, PDiv, PDeath

    Pmove = np.ones((3, gsize))
    Pmove[0,:] = Pmove[0,:]*Pprol
    Pmove[1,:] = Pmove[1,:]*Pprol*25.
    Pmove[2,:] = Pmove[2,:]*Pprol

    PDiv = np.ones((3, gsize))
    PDiv[0,:] = PDiv[0,:]*Pdivi
    PDiv[1,:] = PDiv[1,:]*Pdivi*0.6
    PDiv[2,:] = PDiv[2,:]*0.00

    PDeath = np.ones((3, gsize))
    PDeath[0,:] = PDeath[0,:]*0.00 + np.asarray(np.concatenate(np.where(Ox_irr < O2_crit, 1., 0.)))
    PDeath[1,:] = PDeath[1,:]*Pdie + np.asarray(np.concatenate(np.where(Ox_irr < O2_crit, 1., 0.)))
    PDeath[2,:] = PDeath[2,:]*Pdie*3.0 + np.asarray(np.concatenate(np.where(Ox_irr < O2_crit, 1., 0.)))

    return Pmove, PDiv, PDeath

#=========================================

# Function for creating a more detailed grid around proliferating living cells

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def ext_grid(int gsize, np.ndarray points, np.ndarray Alive, np.ndarray index, np.ndarray Rim, tree, np.ndarray phs, list Discarded, np.ndarray Cell_pop, np.ndarray Dead, np.ndarray GSC, np.ndarray GPP, np.ndarray GDS, double trapped):

    cdef int i, newcomers
    cdef int dsize, nsize
    cdef np.ndarray point, newpoints, xx, Discarded_array
    cdef bint conditions

    # Evaluate new points.
    # NOTE: Only checks around single cells!

    newpoints = phs[np.concatenate(list(tree.query_ball_point(tuple(map(list, points[index[Alive==1]])), np.max([trapped/2.-1.,2]))))]
    nsize = len(newpoints)

    # Create array to store new points based on old grid
    cdef list x_new = [1] * gsize # list(np.ones(gsize))
    cdef list y_new = [1] * gsize 
    for i in range(gsize):
        point = points[i]
        x_new[i] = [np.array(point[0])]
        y_new[i] = [np.array(point[1])]

    dsize = len(Discarded)
    Discarded_array = np.asarray(Discarded)

    # Select only unique points in newpoints

    uni = list(set(tuple(x) for x in newpoints))

    cdef usize = len(uni)

    newcomers = 0

    for i in range(usize):
        xx = np.asarray(uni[i])
        if dsize > 0:
            conditions = (xx not in points)  and (xx not in Discarded_array)
        else:
            conditions = (xx not in points)
        if conditions:
            x_new.append([np.asarray(xx[0])])
            y_new.append([np.asarray(xx[1])])
            newcomers += 1 # How many new points were actually added?

    points = np.append(x_new,y_new,axis=1)

    Rim = np.append(Rim, np.zeros(newcomers))
    Dead = np.append(Dead, np.zeros(newcomers))
    GSC = np.append(GSC, np.zeros(newcomers))
    GPP = np.append(GPP, np.zeros(newcomers))
    GDS = np.append(GDS, np.zeros(newcomers))
    Alive = GSC + GPP + GDS
    Cell_pop = Dead + Alive
    
    # Delaunay trinangulation (Voronoi tesselation) to determine the neighbouring sites

    tri = Delaunay(points)

    # You can later add more points: scipy.spatial.Delaunay.add_points
    # Since the Delaunay triangulation is a numpy object, you can simply delete elements.

    (indptr, indices) = tri.vertex_neighbor_vertices

    gsize = len(points)
    index = np.asarray(range(gsize))

    return x_new, y_new, points, Alive, Rim, index, gsize, tri, indptr, indices, Dead, Cell_pop, GSC, GPP, GDS

#=========================================

# Interpolation functions between grids

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef np.ndarray  interLN( points, np.ndarray values, newpoints):

    cdef np.ndarray nvL, nvN

    nvL = LinearNDInterpolator(points,values)(newpoints)
    nvN = NearestNDInterpolator(points,values)(newpoints)
    nvL[np.isnan(nvL)] = nvN[np.isnan(nvL)]
    return nvL

#=========================================

# Set initial conditions for vasculature and oxygen/nutrient levels

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def init_oxygen(str Restart, np.ndarray Ox_irr, mesh, int low, np.ndarray points):
    if Restart == "None":
        oxygen = CellVariable(mesh = mesh, value = float(low)**2)
    else:
        oxygen = np.concatenate(interLN(points,Ox_irr,(mesh.x,mesh.y)))
        oxygen = CellVariable(mesh=mesh, value=oxygen)
 
    Blood_vessel = CellVariable(mesh=mesh)
    Blood_vessel.setValue(0.)
    Blood_vessel[0] = 500.

    return Blood_vessel, oxygen

#=========================================

# Solve PDE for oxygen

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def solve_oxygen(mesh, oxygen, voronoi_kdtree, np.ndarray pl, np.ndarray GSC, np.ndarray GPP, np.ndarray GDS, Blood_vessel, double Dox, np.ndarray points, list Xs_r, list Ys_r, double timeStepDuration, double lambdaC):

    cdef int j, llen

    llen = len(pl)

    # Figure out how many cells placed in the dense grid lie around the grid points on the mesh on which the PDE is solved.
 
    if np.any(GSC >=1): dist_GSC, region_GSC = voronoi_kdtree.query(points[GSC>=1])
    if np.any(GPP >=1): dist_GPP, region_GPP = voronoi_kdtree.query(points[GPP>=1])
    if np.any(GDS >=1): dist_GDS, region_GDS = voronoi_kdtree.query(points[GDS>=1])
 
    cdef np.ndarray Count_GSC = GSC[GSC>=1]
    cdef np.ndarray Count_GPP = GPP[GPP>=1]
    cdef np.ndarray Count_GDS = GDS[GDS>=1]
 
    GSC_PDE = np.zeros(llen)
    GPP_PDE = np.zeros(llen)
    GDS_PDE = np.zeros(llen)

    for j in range(llen):
        if np.any(GSC >=1): GSC_PDE[j] = np.sum(Count_GSC[region_GSC == j])
        if np.any(GPP >=1): GPP_PDE[j] = np.sum(Count_GPP[region_GPP == j])
        if np.any(GDS >=1): GDS_PDE[j] = np.sum(Count_GDS[region_GDS == j])
 
    GSC_PDE = CellVariable(mesh=mesh, value=GSC_PDE)
    GPP_PDE = CellVariable(mesh=mesh, value=GPP_PDE)
    GDS_PDE = CellVariable(mesh=mesh, value=GDS_PDE)

    # Oxygen differential equation setup with FiPy. 
    # Neumann conditions are automatically adopted as they are left unspecified here.

    eq = TransientTerm() == DiffusionTerm(coeff=Dox) - lambdaC*(GSC_PDE+GPP_PDE+GDS_PDE) + Blood_vessel
    eq.solve(var=oxygen, dt=timeStepDuration)

    # Interpolate to get oxygen concentration on irregular grid.

#    Ox_irr = interLN2(mesh.x,mesh.y,np.asarray(oxygen),Xs_r,Ys_r)
    Ox_irr = interLN((mesh.x,mesh.y),np.asarray(oxygen),(Xs_r,Ys_r))

    return Ox_irr, oxygen

#=========================================

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def hierarchy(int iden, int daughter, np.ndarray GSC, np.ndarray GPP, np.ndarray GDS, double epsilon):

    # Hierarchy for division of tumour cells according to Lan et al. (2017)

    cdef extern from "stdlib.h":
        double drand48()

    cdef double progeny = drand48()

    if iden == 0:
        if progeny <= 1. - epsilon:
            GSC[daughter] += 1.
        else:
            GPP[daughter] += 1.
    elif iden == 1:
        if progeny <= 0.5:
            GPP[daughter] += 1.
        else:
            GDS[daughter] += 1.

    return GSC, GPP, GDS

#=========================================

# Function to write detailed output file with cell_population. Enough information is provided to restart the run.

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def write_snapshot(int tstart, str name, list Input, np.ndarray Ox_irr, np.ndarray points, np.ndarray pl, np.ndarray phs, np.ndarray Cell_pop, np.ndarray GSC, np.ndarray GPP, np.ndarray GDS, np.ndarray Alive, np.ndarray Dead, np.ndarray Rim, list Discarded):

    hf = h5py.File(name,"w")
    hf.create_dataset("time_label", data=tstart)
    hf.create_dataset("Input", data=Input)
    hf.create_dataset('Oxygen', data=Ox_irr)
    hf.create_dataset('Points', data=points)
    hf.create_dataset('points_low', data=pl)
    hf.create_dataset('points_high', data=phs)
    hf.create_dataset('Cell_population', data=Cell_pop)
    hf.create_dataset('GSC_population', data=GSC)
    hf.create_dataset('GPP_population', data=GPP)
    hf.create_dataset('GDS_population', data=GDS)
    hf.create_dataset('Alive_population', data=Alive)
    hf.create_dataset('Dead_population', data=Dead)
    hf.create_dataset('Distance_to_rim', data=Rim)
    hf.create_dataset('Discarded_points', data=Discarded)
    hf.close()

#=========================================

# Function to read snapshots.

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def read_snapshot(name):

    f = h5py.File(name, 'r')
    Input = f['Input'][:]
    Ox_irr = f['Oxygen'][:]
    points = f['Points'][:]
    pl = f['points_low'][:]
    phs = f['points_high'][:]
    Cell_pop = f['Cell_population'][:]
    GSC = f['GSC_population'][:]
    GPP = f['GPP_population'][:]
    GDS = f['GDS_population'][:]
    Alive = f['Alive_population'][:]
    Dead = f['Dead_population'][:]
    Rim = f['Distance_to_rim'][:]
    Discarded = list(f['Discarded_points'][:])
    tstart = f['time_label'][()]

    return Input, Ox_irr, points, pl, phs, Cell_pop, GSC, GPP, GDS, Alive, Dead, Rim, Discarded, tstart

#=========================================

# Initialize run. If a restart file is provided the corresponding snapshot is read in.
# Otherwise, if Restart is "None" a new cell population is computed starting from one GSC near the centre.

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def init_Cell_pop(str Restart, list Input):

    cdef int i, tstart, iden, gsize, resx, resy, steps, SavN, low, Dup, Qup, dx, dy, lx
    cdef double Dox, lambdaC, O2_crit, trapped, epsilon, xx
    cdef bint myplots
    cdef np.ndarray index, Cell_pop, GSC, GPP, GDS, Alive, Dead, Rim, points, Ox_irr
    cdef np.ndarray Ys_rh, Xs_rh, x, y, Xs, Ys, sx, sy, X, Y
    cdef list Discarded, Xs_r, Ys_r

    if Restart == "None": 

        resx, resy, dx, dy, steps, SavN, O2_crit, lambdaC, Dox, low, Dup, trapped, Qup, epsilon, Pdie, Pprol, Pdivi = Input

        mesh = Grid2D(dx=dx, dy=dy, nx=resx, ny=resy)
        x = np.arange(mesh.nx) * mesh.dx
        y = np.arange(mesh.ny) * mesh.dy

        sx = np.random.uniform(0.1,0.9,(resx,resy))
        sy = np.random.uniform(0.1,0.9,(resx,resy))

        # Regular grid
        Y, X = np.meshgrid(y,x)

        # Irregular grid
        Xs = X[:resx-1,:resy-1] + sx[:resx-1,:resy-1]
        Ys = Y[:resx-1,:resy-1] + sy[:resx-1,:resy-1]
        Xs_rh = Xs.reshape(-1,1)
        Ys_rh = Ys.reshape(-1,1)

        # new less resolved grid
        mesh = Grid2D(dx=dx*low, dy=dy*low, nx=resx/low, ny=resy/low)
        Xs_rl = []
        Ys_rl = []
        lx = len(mesh.x)
        for i in range(lx):
            xx = mesh.x[i]
            j = (abs(Xs_rh-xx)**2 + abs(Ys_rh-np.asarray(mesh.y[i]))**2).argmin()
            Xs_rl.extend([Xs_rh[j]])
            Ys_rl.extend([Ys_rh[j]])

        Ox_irr = np.asarray([])

        points = np.append(Xs_rl,Ys_rl,axis=1)

        pl = points.copy()

        phs =  np.c_[Xs.ravel(), Ys.ravel()]

        Discarded = []
 
        gsize = len(points)
        index = np.asarray(range(gsize))
 
        # Spawn one cell (close to the centre of the grid)    
        Cell_pop = np.zeros(gsize) # All cells dead or alive
        Cell_pop[round(resx*resy/low**2*0.5-resy/2/low)] = 1
        GSC = Cell_pop.copy()   # Stem cells
        GPP = np.zeros(gsize)   # Proliferative progeny
        GDS = np.zeros(gsize)   # Differenting subpopulation
        Alive = Cell_pop.copy() # Alive cells
        Dead = np.zeros(gsize)  # Dead cells
        Rim = np.zeros(gsize)   # Distance to rim
        tstart = -1

    else:

        Input, Ox_irr, points, pl, phs, Cell_pop, GSC, GPP, GDS, Alive, Dead, Rim, Discarded, tstart = read_snapshot(Restart)
        resx, resy, dx, dy, steps, SavN, O2_crit, lambdaC, Dox, low, Dup, trapped, Qup = Input
        mesh = Grid2D(dx=dx*low, dy=dy*low, nx=resx/low, ny=resy/low)
        gsize = len(points)
        index = np.asarray(range(gsize))

    # Tree to find nearest neighbours in low-resolution grid
    voronoi_kdtree = cKDTree(pl)

    # Tree to find nearest neighbours in dense grid
    tree = cKDTree(phs)

    return gsize, tstart, Input, mesh, Ox_irr, low, points, pl, phs, index, Cell_pop, GSC, GPP, GDS, Alive, Dead, Rim, Discarded, voronoi_kdtree, tree

#=========================================

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def run_ABM(str Restart, str store, int resx, int resy, int steps, int SavN, double Dox, double lambdaC, double O2_crit, int low, int Dup, double trapped, int Qup, double epsilon, double Pdie, double Pprol, double Pdivi,  double dx, double dy, bint myplots):

    cdef double start_time = time.time()
    cdef str cwd = os.getcwd()

    # Cython definitions
    cdef str name, logfile, paramsfile
    cdef int i, j, k
    cdef int Cell, tstart, new, daughter, gsize, upper, idummy1, idummy2, idummy3, idummy4, idummy5, asize, a_tot
    cdef float Reaper, move, divide, Pup, fdummy1, fdummy2, fdummy3
    cdef np.ndarray neigh, index, Cell_pop, GSC, GPP, GDS, Alive, Dead, Rim, points, Ox_irr, Pmove, PDiv, PDeath, free, mask
    cdef list Discarded, Proliferating, Quiescent, Xs_r, Ys_r
    cdef bint DIV
    cdef int iden = 0

    cdef extern from "stdlib.h":
        double drand48()

    cdef list Input = [resx, resy, dx, dy, steps, SavN, O2_crit, lambdaC, Dox, low, Dup, trapped, Qup, epsilon, Pdie, Pprol, Pdivi]

    gsize, tstart, Input, mesh, Ox_irr, low, points, pl, phs, index, Cell_pop, GSC, GPP, GDS, Alive, Dead, Rim, Discarded, voronoi_kdtree, tree = init_Cell_pop(Restart, Input)

    Blood_vessel, oxygen = init_oxygen(Restart, Ox_irr, mesh, low, points)

    # Time-step, to be defined. For reaction-diffusion equations
    # timeStepDuration = 10 * 0.9 * dx**2 / (2 * Dox)
    cdef float timeStepDuration = 4.5
  
    if myplots:
        plt.figure(1,figsize=(8,8))
        plt.figure(2,figsize=(8,8))
        GSC_tot = []
        GPP_tot = []
        GDS_tot = []
        Dead_tot = []
        Alive_single = []

    # Save summary of history
    logfile = cwd + "/" + store + "/history.log"
    if not os.path.isfile(logfile) or tstart == -1:
        ff = open(logfile, "w")
        ff.write("# Summary file for ABM. \n") 
        ff.write("# Time-step, runtime, GSC, GPP, GDS, Dead, Single cells \n")
        ff.close()

    # Save model parameters for run
    paramsfile = cwd + "/" + store + "/params.log"
    if not os.path.isfile(paramsfile) or tstart == -1:
        ff = open(paramsfile, "w")
        ff.write("# resx, resy, steps, SavN, Dox, lambdaC, O2_crit, low, Dup, trapped, Qup, epsilon, Pdie, Pprol, Pdivi, dx, dy \n")
        ff.write("%i %i %i %i %.9E %.9E %.9E %i %i %.9E %i %.9E %.9E %.9E %.9E %.2E %.2E \n" % (resx, resy, steps, SavN, Dox, lambdaC, O2_crit, low, Dup, trapped, Qup, epsilon, Pdie, Pprol, Pdivi, dx, dy))
        ff.close()

    print("Grid creation: --- %s seconds ---" % (time.time() - start_time))

    for i in range(tstart+1,steps):
 
        # 1) Extend grid for cells based on the positions of the current cell population

#        t1 = time.time()

        asize = len(Alive[Alive==1])

        if asize > 0:
            Xs_r, Ys_r, points, Alive, Rim, index, gsize, tri, indptr, indices, Dead, Cell_pop, GSC, GPP, GDS = ext_grid(gsize, points, Alive, index, Rim, tree, phs, Discarded, Cell_pop, Dead, GSC, GPP, GDS, trapped)

#        print("Extending grid: %s seconds" % (time.time() - t1))

#        t1 = time.time()

        # 2) Solve PDE for oxygen

        Ox_irr, oxygen = solve_oxygen(mesh, oxygen, voronoi_kdtree, pl, GSC, GPP, GDS, Blood_vessel, Dox, points, Xs_r, Ys_r, timeStepDuration, lambdaC)

        # 3) Evaluate propabilities at each mesh point

        Pmove, PDiv, PDeath = stocev(Ox_irr, O2_crit, Pdie, Pprol, Pdivi)

#        print("Oxygen: %s seconds" % (time.time() - t1))

#        t1 = time.time()

        # 4) Consider all single cells
        # Go through the cells in random order
        Proliferating = sorted(index[Alive==1], key=lambda k: random.random())

        for k in Proliferating:

            if GSC[k] == 1.: iden = 0
            if GPP[k] == 1.: iden = 1
            if GDS[k] == 1.: iden = 2

            # Apoptosis?
            Reaper = drand48()
            Pup = PDeath[iden, k]
            if Reaper <= Pup:
                Alive[k] = 0.
                GSC[k] = 0. 
                GPP[k] = 0.
                GDS[k] = 0.
                Dead[k] += 1.
                Rim[k] = 2*trapped
            # Is the cell proliferative or trapped?

            elif Rim[k] < trapped:

                DIV = False
                neigh = indices[indptr[k]:indptr[k+1]]
                neigh = neigh[Cell_pop[neigh] == 0]

                # Division to empty neighbouring spot
                if len(neigh) > 0:
                    divide = drand48()
                    Pup = PDiv[iden, k]
                    if divide <= Pup:
                        fdummy1 = len(neigh)
                        daughter = int(fdummy1*drand48())
                        daughter = neigh[daughter]
                        Alive[daughter] = 1.
                        Cell_pop[daughter] = 1.
                        GSC, GPP, GDS = hierarchy(iden, daughter, GSC, GPP, GDS, epsilon)
                        DIV = True # If you divided you don't move 

                # Otherwise, division is allowed, if the neighbours can be pushed aside
                else:
                    free = points[Cell_pop==0]
                    idummy1 = int(len(free)) # If there are no free spots at all, there is no need to continue here. Indeed, np.where would crash.
                    if idummy1 > 0:
                        kdtree = cKDTree(free)
                        res = kdtree.query(points[k])
                        fdummy1 = res[0]
                        Rim[k] = fdummy1
                        divide = drand48()
                        daughters = np.where(points == free[res[1]])
                        daughter  = daughters[0][0]
                        fdummy2 = PDiv[iden, k]
                        fdummy2 = fdummy2*c_exp(-fdummy1/trapped/2.)

                        # Note: It gets harder to push the further we get from the rim, hence exponential damping 
                        if fdummy1 < trapped and fdummy1 > 0. and divide <= fdummy2:
                            Alive[daughter] = 1.
                            Cell_pop[daughter] = 1.
                            GSC, GPP, GDS = hierarchy(iden, daughter, GSC, GPP, GDS, epsilon)
                            DIV = True

                # Movement only if no division and free spot next to cell
                if DIV == False and len(neigh) > 0:
                    move = drand48()
                    Pup = Pmove[iden,k]
                    if move <= Pup:
                        # Move away from point
                        Alive[k] = 0.
                        Cell_pop[k] = 0.
                        if iden == 0: GSC[k] = 0. 
                        if iden == 1: GPP[k] = 0.
                        if iden == 2: GDS[k] = 0.
                        # To random free neighbour
#                        new = random.choice(neigh)
                        fdummy1 = len(neigh)
                        new = int(fdummy1*drand48())
                        new = neigh[new]
                        Alive[new] = 1.
                        Cell_pop[new] = 1.
                        if iden == 0: GSC[new] = 1.
                        if iden == 1: GPP[new] = 1.
                        if iden == 2: GDS[new] = 1.

#        print("Single cell: %s seconds " % (time.time() - t1))

#        t1 = time.time()

        # 5) Consider all clusters with cells that are alive.
        Quiescent = sorted(index[Alive>1], key=lambda k: random.random())
 
        for k in Quiescent:
            # Do the cells die of oxygen shortage?
            if all(PDeath[:,k] >= 1.):
                Dead[k] += Alive[k]
                Alive[k] = 0.
                GSC[k] = 0. 
                GPP[k] = 0. 
                GDS[k] = 0.
            else:
                upper = int(GSC[k])
                for Cell in range(upper):
                    Reaper = drand48()
                    Pup = PDeath[0, k]
                    if Reaper <= Pup:
                        Alive[k] -= 1.
                        GSC[k] -= 1.
                        Dead[k] += 1.
                upper = int(GPP[k])
                for Cell in range(upper):
                    Reaper = drand48()
                    Pup = PDeath[1, k]
                    if Reaper <= Pup:
                        Alive[k] -= 1.
                        GPP[k] -= 1.
                        Dead[k] += 1.
                upper = int(GDS[k])
                for Cell in range(upper):
                    Reaper = drand48()
                    Pup = PDeath[2, k]
                    if Reaper <= Pup:
                        Alive[k] -= 1.
                        GDS[k] -= 1.
                        Dead[k] += 1.

#        print("Clusters: %s seconds" % (time.time() - t1))

#        t1 = time.time()

        # 6) Delete points around clusters of dead or trapped cells

        a_tot = int(np.sum(Alive[Alive>=1])) # There is no need to resort, if all cells are dead

        if a_tot > 0:
            gsize, GSC, GPP, GDS, Alive, Cell_pop, Dead, Rim, Xs_r, Ys_r, Discarded, points, tri, index, indptr, indices = Dead_cluster(GSC, GPP, GDS, Alive, Cell_pop, Dead, Rim, Xs_r, Ys_r, Discarded, points, tri, index, indptr, indices, Dup, Qup, trapped)

#        print("Redistribute: %s seconds" % (time.time() - t1))

#        t1 = time.time()

        # 7) Save snapshot if required and save data for overview

        mask = Alive == 1
        mask = np.logical_and(mask, Cell_pop == 1.)

        if myplots:
            GSC_tot.extend([np.sum(GSC)])
            GPP_tot.extend([np.sum(GPP)])
            GDS_tot.extend([np.sum(GDS)])
            Dead_tot.extend([np.sum(Dead)])
            Alive_single.extend([len(Cell_pop[mask])])

        # Write log-file
        idummy1 = int(np.sum(GSC))
        idummy2 = int(np.sum(GPP))
        idummy3 = int(np.sum(GDS))
        idummy4 = int(np.sum(Dead))
        idummy5 = len(Cell_pop[mask])

        fdummy1 = time.time() - start_time
        ff = open(logfile, "a")
        ff.write("%i %.2E %i %i %i %i %i \n" % (i, fdummy1,  idummy1, idummy2, idummy3, idummy4, idummy5))
        ff.close()

        if np.mod(i,SavN) == 0:
  
            exten = "%05d" % i

            if myplots:
                plt.figure(1)
                plt.clf()
                plt.triplot(points[:,0], points[:,1], tri.simplices,color="c",alpha=0.5)
                for k in index[Dead>0]:
                    plt.plot(points[k][0],points[k][1],"k.")
                for k in index[GSC==1]:
                    plt.plot(points[k][0],points[k][1],".",color="m")
                for k in index[GPP==1]:
                    plt.plot(points[k][0],points[k][1],"r.")
                for k in index[GDS==1]:
                    plt.plot(points[k][0],points[k][1],"y.")
                for k in index[Alive > 0]:
                    if Cell_pop[k] > 1:
                        plt.plot(points[k][0],points[k][1],".",color="blue")
                plt.xlabel(r'$\mathrm{x\, lattice \, position}$',fontsize=16)
                plt.ylabel(r"$\mathrm{y\, lattice \, position}$",fontsize=16)
                plt.xticks(size=16)
                plt.yticks(size=16)
                plt.gcf().subplots_adjust(bottom=0.18)
                plt.gcf().subplots_adjust(left=0.18)

                plt.savefig(cwd+"/"+store+"/Snapshot_"+exten+".png", bbox_inches='tight')

            print("Cell population of %d evolves. %d are modelled individually. In addition, %d are dead. Total population: %d." % (np.sum(Alive[Alive>=1]),len(Alive[mask]),np.sum(Dead[Dead>=1]),np.sum(Cell_pop)))
            print("Time-step: %d --- %s seconds ---" % (i , time.time() - start_time))
            name = cwd+"/"+store+"/Data_"+exten+".h5"
            Ox_irr = interLN((mesh.x,mesh.y),np.asarray(oxygen),(Xs_r,Ys_r))
            write_snapshot(i, name, Input, Ox_irr, points, pl, phs, Cell_pop, GSC, GPP, GDS, Alive, Dead, Rim, Discarded)

            if myplots:
                plt.figure(2)
                plt.semilogy(GSC_tot,color="m",linestyle="--",label=r"$\mathrm{GSC}$")
                plt.semilogy(GPP_tot,color="r",linestyle="-.",label=r"$\mathrm{GPP}$")
                plt.semilogy(GDS_tot,color="y",linestyle="-",label=r"$\mathrm{GDS}$")
                plt.semilogy(np.asarray(GSC_tot)+np.asarray(GPP_tot)+np.asarray(GDS_tot),color="g",linestyle="-",label=r"$\mathrm{Alive}$")
                plt.semilogy(Alive_single,color="orange",linestyle="--",label=r"$\mathrm{Alive, \, Rim}$",dashes=(5, 5))
                plt.semilogy(Dead_tot,color="k",linestyle="-.",label=r"$\mathrm{Dead}$")
                if i == tstart+1:
                    plt.legend(fontsize=16)
                plt.xlabel(r'$t$',fontsize=16)
                plt.ylabel(r"$\mathrm{Cell \, number}$",fontsize=16)
                plt.xticks(size=16)
                plt.yticks(size=16)
                plt.gcf().subplots_adjust(bottom=0.18)
                plt.gcf().subplots_adjust(left=0.18)
                plt.savefig(cwd+"/"+store+"/Population_overview.png",bbox_inches='tight')

#        print("Saving: %s seconds " % (time.time() - t1))

#        t1 = time.time()

