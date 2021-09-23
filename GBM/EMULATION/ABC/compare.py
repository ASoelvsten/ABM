import numpy as np
from glob import glob
import tensorflow
import pickle
import pygtc
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
from scipy.stats import wasserstein_distance as wd
import os.path
import time

def MAXL(p):
    p, bins, patches = plt.hist(p, bins =50, density=True)
    plt.close()
    MLE = bins[p.argmax()]
    return MLE

def summa(i, lam, LAM, lambdaC):
    lam_median = np.percentile(lambdaC,50)
    lam_std1 = np.percentile(lambdaC,16)
    lam_std2 = np.percentile(lambdaC,84)
    lam_std3 = np.percentile(lambdaC,5)
    lam_std4 = np.percentile(lambdaC,95)
    lam_mean = np.mean(lambdaC)
    MLE = MAXL(lambdaC)

    print(lam, lam_median, lam_mean, MLE)

    res_lam = [lam, lam_median, lam_mean, MLE, lam_std1, lam_std2, lam_std3, lam_std4]
 
    if len(LAM) < 1:
        LAM = res_lam
    else:
        LAM = np.vstack((LAM,res_lam))

    return LAM

starttime = time.time()
ptime = time.process_time()

data = sorted(glob("../../DIST/summary_*.txt"))

LAM = []
DIV = []
EPS = []
DIE = []

num = len(data)

TEST = False

Failures = 0

for jot, obs_name in enumerate(data):

    f = data[jot]

    eps = float(f[f.find("E_")+2:f.find("_L_")])
    lam = float(f[f.find("_L_")+3:f.find("_D_")])
    div = float(f[f.find("_D_")+3:f.find("_cd_")])
    die = float(f[f.find("_cd_")+4:f.find(".txt")])

    print("lam, eps, die, div: ", lam, eps, die, div)

    name = "./E_"+str(eps)+"_L_"+str(lam)+"_D_"+str(div)+"_cd_"+str(die)

    sam_file = "./" + name + "_sam_emu.txt"

    if os.path.isfile(sam_file):

        print("Running %i of %i ..." % (jot+1, num))
        print(sam_file)

        result = np.genfromtxt(sam_file)

        obs = np.genfromtxt(f)

        CI = np.percentile(obs,84,axis=0)-np.percentile(obs,16,axis=0)

        if 0.0 not in CI:

            print(result)
            print("Max", np.max(np.concatenate(result)))

            epsilon = result[:,0]
            lambdaC = result[:,1]
            Pdiv = result[:,2]
            Pdie = result[:,3]

            LAM = summa(0, lam, LAM, lambdaC)
            EPS = summa(1, eps, EPS, epsilon)
            DIE = summa(2, die, DIE, Pdie)
            DIV = summa(3, div, DIV, Pdiv)
 
            if TEST:
                truths = ((lam, eps, die, div))

                names = [
                '$\lambda_\mathrm{C}$',
                '$\epsilon$',
                '$P_\mathrm{cd}$',
                '$P_\mathrm{div}$',
                    ]

                chainLabels = ['Neural Network']
                truthLabels = ('Truth')

                # Do the magic
                GTC = pygtc.plotGTC(chains=[np.column_stack((lambdaC,epsilon,Pdie,Pdiv))],
                figureSize='MNRAS_page',
                truths = truths,
                paramNames=names,
                plotName= name+'_ABC.pdf',
                legendMarker='All',
                )

#                plt.show()
        else:
            print("Failed")
            Failures += 1

print("#Failures: ", Failures)

Res = np.column_stack((LAM,EPS,DIV,DIE))
np.savetxt("Res_emu_abc.txt",Res)

print("TIMING")
print(time.time()-starttime)
print(time.process_time()-ptime)
