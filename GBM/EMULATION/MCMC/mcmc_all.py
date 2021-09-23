import numpy as np
import emcee
import pygtc
import pickle
import matplotlib.pyplot as plt
import time
import tensorflow
from multiprocessing import Pool
from multiprocessing import cpu_count
import os
import sys
import shutil
import random
from glob import glob
import time

starttime = time.time()
ptime = time.process_time()

Burn_sims = 5000
Reg_sims  = 20000

Learner = "GP" # "GP" or "Neural"
L_type = "A" # "A" or "B"

#=========================================================================

if Learner == "GP":

    ext = "104_clean_RBF"
    
    with open("../emu_GP_model_"+ext+".pkl", 'rb') as run:
        model = pickle.load(run)

    with open("../emu_GP_sc_"+ext+".pkl", 'rb') as run:
        sc = pickle.load(run)

    with open("../emu_GP_scy_"+ext+".pkl", 'rb') as run:
        scy = pickle.load(run)

    multiple = True

elif Learner == "Neural":

    ext = "104_clean"

    model = tensorflow.keras.models.load_model("../emu_model_"+ext+".h5", compile=False)

    with open("../emu_sc_"+ext+".pkl", 'rb') as run:
        sc = pickle.load(run)

    with open("../emu_scy_"+ext+".pkl", 'rb') as run:
        scy = pickle.load(run)

    multiple = False

#=========================================================================

def log_prob(params, y_obs, std_obs):

    if any(params < 0) or any(params > 0.5):
        return -np.inf
    else:
        if Learner == "Neural":
            if L_type == "A":
                X_Dist = [[params[0],params[1],params[2],params[3]]]
                X_Dist = sc.transform(X_Dist)
                y_Dist = model.predict(X_Dist)
                y_Dist = y_Dist[0][:5]
                y_Dist = scy.inverse_transform(y_Dist)
                Likelihood = -1.0*np.sum((y_obs-y_Dist)**2/(std_obs**2))
            elif L_type == "B":
                X_Dist = [params[0],params[1],params[2],params[3]]
                X_D = [params[0],params[1],params[2],params[3]]
                for i in range(99):
                    X_Dist = np.vstack((X_Dist, X_D))
                X_Dist = sc.transform(X_Dist)
                y_D = model.predict(X_Dist)
                y_Dist = y_D[:,:5]
                y_Dist = scy.inverse_transform(y_Dist)
                y_pred = np.median(y_Dist,axis=0)
                std_pred = np.std(y_Dist,axis=0)
                Likelihood = -1.0*np.sum((y_obs-y_pred)**2/(std_pred**2+std_obs**2))
        elif Learner == "GP":
            if L_type == "A":
                X_Dist = [[params[0],params[1],params[2],params[3]]]
                X_Dist = sc.transform(X_Dist)
                y_Dist = model.sample_y(X_Dist,random_state=None)
                y_Dist = np.transpose(y_Dist[0][:5])
                y_Dist = scy.inverse_transform(y_Dist)
                Likelihood = -1.0*np.sum((y_obs-y_Dist)**2/std_obs**2)
            elif L_type == "B":
                X_Dist = [[params[0],params[1],params[2],params[3]]]
                X_Dist = sc.transform(X_Dist)
                y_Dist = model.sample_y(X_Dist,100,random_state=None)
                y_Dist = np.transpose(y_Dist[0][:,:5])
                y_Dist = scy.inverse_transform(y_Dist)
                Likelihood = -1.0*np.sum((y_obs-np.median(y_Dist,axis=0))**2/(std_obs**2+np.std(y_Dist,axis=0)**2))

        return Likelihood

#=========================================================================

files = sorted(glob("../../DIST/summar*"))

print("Number of cases: ", len(files))

start = 0

for i, f in enumerate(files):

    if i >= start:

        eps = float(f[f.find("E_")+2:f.find("_L_")])
        lam = float(f[f.find("_L_")+3:f.find("_D_")])
        div = float(f[f.find("_D_")+3:f.find("_cd_")])
        die = float(f[f.find("_cd_")+4:f.find(".txt")])

        truths = np.asarray([lam, eps, die, div])

        name = "E_"+str(truths[1])+"_L_"+str(truths[0])+"_D_"+str(truths[3])+"_cd_"+str(truths[2])

        fname = name +"_MCMC_"+Learner+"_summary.txt"
        fnameb = name +"_MCMC_"+Learner+"_burnin.txt"

        if os.path.isfile(fname):
            print("Done")
        else:
            obs = np.genfromtxt(files[i])

            y_obs = [[np.median(obs[:,0]),np.median(obs[:,1]),np.median(obs[:,2]),np.median(obs[:,3]),np.median(obs[:,4])]]

            std_obs = np.std(obs,axis=0)

            print("OBS:", y_obs,std_obs)

            ndim =4
            nwalkers = 2*ndim

            p0 = np.random.uniform(low=0.0, high=0.5, size=(nwalkers,ndim))

            if 0.0 not in std_obs:
                if multiple == True:

                    with Pool() as pool:
                        print("Running with multiple cores")
                        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[y_obs,std_obs], pool=pool)
                        burnin = sampler.run_mcmc(p0, Burn_sims)
                        samples_burnin = sampler.get_chain(flat=True)
                        np.savetxt(fnameb,samples_burnin)
                        print("Burnin over")
                        sampler.reset()
                        sampler.run_mcmc(burnin, Reg_sims)

                    ncpu = cpu_count()
                    print("{0} CPUs".format(ncpu))

                else:
                    print("Running with single core")
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[y_obs,std_obs])
                    burnin = sampler.run_mcmc(p0, Burn_sims)
                    samples_burnin = sampler.get_chain(flat=True)
                    np.savetxt(fnameb,samples_burnin)
                    print("Burnin over")
                    sampler.reset()
                    sampler.run_mcmc(burnin, Reg_sims)

                samples = sampler.get_chain(flat=True)
  
                np.savetxt(fname,samples)

                names = [
                    '$\lambda_\mathrm{C}$',
                    '$\epsilon$',
                    '$P_\mathrm{cd}$',
                    '$P_\mathrm{div}$',
                    ]

                truthLabels = ('Truth')

                # Do the magic
                GTC = pygtc.plotGTC(chains=[samples],
                figureSize='MNRAS_page',
                truths = truths,
                paramNames=names,
                plotName=name+"_MCMC_"+Learner+".png",
                truthLabels=truthLabels,
                legendMarker='All',
                )
            else:
                print("0.0 in std")

print("TIMING")
print(time.time()-starttime)
print(time.process_time()-ptime)
