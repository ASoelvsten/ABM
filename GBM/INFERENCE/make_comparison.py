import numpy as np
from glob import glob
import tensorflow
import pickle
import pygtc
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
from scipy.stats import wasserstein_distance as wd
import os.path
from sklearn.neighbors import KernelDensity
import time

ext = "103_clean"
ML = "GP" # NN or GP

#====================================================================

def MAXL(p):
    p, bins, patches = plt.hist(p, bins =50, density=True)
    plt.close()
    MLE = bins[p.argmax()]
    return MLE

#====================================================================

def QTR(q,truth):
    X = q.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(X)
    qtr = kde.score_samples(np.asarray(truth).reshape(-1, 1))
    return -qtr[0]
  
#===================================================================
# Create array with summary statistics of data and prediction

def summa(i, lam, LAM, y_Dist, samples):
    lam_median = np.percentile(y_Dist[:,i],50)
    lam_std1 = np.percentile(y_Dist[:,i],84) - np.percentile(y_Dist[:,i],16)
    lam_std2 = np.std(y_Dist[:,i])
    lam_std3 = np.percentile(y_Dist[:,i],5)
    lam_std4 = np.percentile(y_Dist[:,i],95)
    lam_mean = np.mean(y_Dist[:,i])
    MLE = MAXL(y_Dist[:,i])

    print("DIST STD:", lam_std1/2.,lam_std2)

    s_median = np.percentile(samples[:,i],50)
    sstd1 = np.percentile(samples[:,i],84) - np.percentile(samples[:,i],16)
    sstd2 = np.std(samples[:,i])
    sstd3 = np.percentile(samples[:,i],5)
    sstd4 = np.percentile(samples[:,i],95)
    s_mean = np.mean(samples[:,i])

    print(lam, lam_median, lam_mean, MLE)

    qtr = QTR(y_Dist[:,i],lam)

    if i < 4:
        res_lam = [lam, s_median, s_mean, sstd1, sstd2, sstd3, sstd4, lam_median, lam_mean, MLE, lam_std1, lam_std2, lam_std3, lam_std4,qtr]
    else:
        s_medianb = np.percentile(samples[:,i+1],50)
        sstd1b = np.percentile(samples[:,i+1],84) - np.percentile(samples[:,i+1],16)
        sstd2b = np.std(samples[:,i+1])
        sstd3b = np.percentile(samples[:,i+1],5)
        sstd4b = np.percentile(samples[:,i+1],95)
        s_meanb = np.mean(samples[:,i+1])

        res_lam = [lam, s_median, s_mean, sstd1, sstd2, sstd3, sstd4, lam_median, lam_mean, MLE, lam_std1, lam_std2, lam_std3, lam_std4, s_medianb, sstd1b, sstd2b, sstd3b, sstd4b, s_meanb, qtr]
 
    if len(LAM) < 1:
        LAM = res_lam
    else:
        LAM = np.vstack((LAM,res_lam))

    return LAM

#====================================================================

starttime = time.time()
ptime = time.process_time()

data = glob("../DIST/summary_*.txt")

LAM = []
DIV = []
EPS = []
DIE = []

TEST = False

num = len(data)

print(ext)

if ML == "NN":

    model = tensorflow.keras.models.load_model("infe_model_"+ext+".h5", compile=False)

    with open("infe_sc_"+ext+".pkl", 'rb') as run:
        sc = pickle.load(run)

    with open("infe_scy_"+ext+".pkl", 'rb') as run:
        scy = pickle.load(run)

elif ML == "GP":

    with open("infe_GP_model_"+ext+".pkl", 'rb') as run:
        model = pickle.load(run)

    with open("infe_GP_sc_"+ext+".pkl", 'rb') as run:
        sc = pickle.load(run)

    with open("infe_GP_scy_"+ext+".pkl", 'rb') as run:
        scy = pickle.load(run)

for j, f in enumerate(data):

    eps = float(f[f.find("E_")+2:f.find("_L_")])
    lam = float(f[f.find("_L_")+3:f.find("_D_")])
    div = float(f[f.find("_D_")+3:f.find("_cd_")])
    die = float(f[f.find("_cd_")+4:f.find(".txt")])

    print("lam, eps, die, div: ", lam, eps, die, div)

    name = "E_"+str(eps)+"_L_"+str(lam)+"_D_"+str(div)+"_cd_"+str(die)

    if os.path.isfile("../DIST/summary_"+name+".txt"):

        print("Running %i of %i ..." % (j+1, num))

        name = "E_"+str(eps)+"_L_"+str(lam)+"_D_"+str(div)+"_cd_"+str(die)

        samples = np.genfromtxt("../DIST/summary_"+name+".txt")

        obs = []
        std = []

        for i in range(5):
            s_median = np.percentile(samples[:,i],50)
            obs.extend([s_median])
            std.extend([np.std(samples[:,i])])

        print("OBS: ", obs, std)

        CI = np.percentile(samples,84,axis=0)-np.percentile(samples,16,axis=0)

        if 0.0 not in CI:

            dim = len(obs)
            cov = list(np.zeros(dim))
            cov[0] = std[0]**2

            for i in range(dim):
                if i >= 1:
                    newrow = list(np.zeros(dim))
                    cov = np.vstack((cov, newrow))
                    cov[i,i] = std[i]**2

            if ML == "NN":

                samples_x = 10000

                X_Dist = np.random.multivariate_normal(obs, cov, samples_x)

                X_Dist = sc.transform(X_Dist)

                y_Dist = model.predict(X_Dist)

                y_Dist = y_Dist[:,:4]
 
                y_Dist = scy.inverse_transform(y_Dist)
 
                LAM = summa(0, lam, LAM, y_Dist, samples)
                EPS = summa(1, eps, EPS, y_Dist, samples)
                DIE = summa(2, die, DIE, y_Dist, samples)
                DIV = summa(3, div, DIV, y_Dist, samples)

            if ML == "GP":
                
                samples_x = 100

                X_Dist = np.random.multivariate_normal(obs, cov, samples_x)

                X_Dist = sc.transform(X_Dist)

                print("Sampling")

                y = model.sample_y(X_Dist,100,random_state=None)

                y_D = []

                for i, j in enumerate(y):
                    j = np.transpose(j)
                    if len(y_D) < 1:
                        y_D = j
                    else:
                        y_D = np.vstack((y_D,j))

                y_Dist = scy.inverse_transform(y_D)

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
                GTC = pygtc.plotGTC(chains=[y_Dist],
                figureSize='MNRAS_page',
                truths = truths,
                paramNames=names,
                legendMarker='All',
                )

                plt.show()
 
Res = np.column_stack((LAM,EPS,DIV,DIE))
np.savetxt("Res_infe_"+ext+".txt",Res)

print(ext)
print(time.time()-starttime)
print(time.process_time()-ptime)
