import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import time

starttime = time.time()
ptime = time.process_time()

ext = "_Neural_" # "_GP_" "_Neural_"

#=========================================================================

def QTR(q,truth):
    X = q.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(X)
    qtr = kde.score_samples(np.asarray(truth).reshape(-1, 1))
    return -qtr[0]

#=========================================================================

def read_MCMC(fil):

    data = np.genfromtxt(fil)
    eps = float(fil[fil.find("E_")+2:fil.find("_L_")])
    lam = float(fil[fil.find("_L_")+3:fil.find("_D_")])
    div = float(fil[fil.find("_D_")+3:fil.find("_cd_")])
    die = float(fil[fil.find("_cd_")+4:fil.find("_MCMC_")])

    print(eps, lam, div, die)

    EPS = data[:,1]
    LAM = data[:,0]
    DIV = data[:,3]
    DIE = data[:,2]

    qtr_lam = QTR(LAM, lam)
    qtr_eps = QTR(EPS, eps)
    qtr_div = QTR(DIV, div)
    qtr_die = QTR(DIE, die)

    MMD_lam = np.mean(LAM)-lam
    MMD_eps = np.mean(EPS)-eps
    MMD_div = np.mean(DIV)-div
    MMD_die = np.mean(DIE)-die

    mmd_lam = abs(np.mean(LAM)-lam)/lam # np.percentile(LAM,50)-lam
    mmd_eps = abs(np.mean(EPS)-eps)/eps # np.percentile(EPS,50)-eps
    mmd_div = abs(np.mean(DIV)-div)/div # np.percentile(DIV,50)-div
    mmd_die = abs(np.mean(DIE)-die)/die # np.percentile(DIE,50)-die

    ssd_lam = np.std(LAM)
    ssd_eps = np.std(EPS)
    ssd_div = np.std(DIV)
    ssd_die = np.std(DIE)

    return [MMD_lam,MMD_eps,MMD_div,MMD_die,mmd_lam,mmd_eps,mmd_div,mmd_die,qtr_lam,qtr_eps,qtr_div,qtr_die,lam,eps,div,die,ssd_lam,ssd_eps,ssd_div,ssd_die]

#=========================================================================

files = glob("../../DIST/summar*")

print("Number of cases: ", len(files))

Res = []

Done = 0
Failed = 0

for i, f in enumerate(files):

    eps = float(f[f.find("E_")+2:f.find("_L_")])
    lam = float(f[f.find("_L_")+3:f.find("_D_")])
    div = float(f[f.find("_D_")+3:f.find("_cd_")])
    die = float(f[f.find("_cd_")+4:f.find(".txt")])

    truths = np.asarray([lam, eps, die, div])

    name = "E_"+str(truths[1])+"_L_"+str(truths[0])+"_D_"+str(truths[3])+"_cd_"+str(truths[2])
    fname = name+"_MCMC"+ext+"summary.txt"

    if os.path.isfile(fname):
        print(name)
        obs = np.genfromtxt("../../DIST/summary_"+name+".txt")

        CI = np.percentile(obs,84,axis=0)-np.percentile(obs,16,axis=0)

        if 0.0 not in CI:
            print("Sim. %i worked fine" %(i))
            res = read_MCMC(fname)
            Done += 1 
            if len(Res) == 0:
                Res = res
            else:
                Res = np.vstack((Res,res))
        else:
            print("0.0 in std")
            Failed += 1

print(Res[:,8:12])

print("Worked and failed: ", Done, Failed, Done+Failed)

np.savetxt("Res_MCMC_emu"+ext+"104_clean.txt",Res)

print("TIMING")
print(time.time()-starttime)
print(time.process_time()-ptime)

