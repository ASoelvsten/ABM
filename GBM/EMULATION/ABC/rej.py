# Regression Example With Boston Dataset: Baseline
from pandas import read_csv
import h5py
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
# load dataset
import elfi
import pickle
import tensorflow
from glob import glob
import sys

jot = int(sys.argv[1])

ext = "104_clean"
ML = "NN" # NN, GP

storage = "./"
N = 10000
qs = 0.001

#========================================================

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

data = sorted(glob("../../DIST/summary_*.txt"))

f = data[jot]

eps = float(f[f.find("E_")+2:f.find("_L_")])
lam = float(f[f.find("_L_")+3:f.find("_D_")])
div = float(f[f.find("_D_")+3:f.find("_cd_")])
die = float(f[f.find("_cd_")+4:f.find(".txt")])

print("lam, eps, die, div: ", lam, eps, die, div)

name = "E_"+str(eps)+"_L_"+str(lam)+"_D_"+str(div)+"_cd_"+str(die)

samples = np.genfromtxt("../../DIST/summary_"+name+".txt")

print(len(data),len(samples))

y_obs = []
std = []

for i in range(5):
    s_median = np.percentile(samples[:,i],50)
    y_obs.extend([s_median])
    std.extend([np.std(samples[:,i])])

if ML == "NN":
    model = tensorflow.keras.models.load_model("../emu_model_"+ext+".h5", compile=False)

    with open("../emu_sc_"+ext+".pkl", 'rb') as run:
        sc = pickle.load(run)

    with open("../emu_scy_"+ext+".pkl", 'rb') as run:
        scy = pickle.load(run)

elif ML == "GP":

    with open("../emu_GP_model_"+ext+".pkl", 'rb') as run:
        model = pickle.load(run)

    with open("../emu_GP_sc_"+ext+".pkl", 'rb') as run:
        sc = pickle.load(run)

    with open("../emu_GP_scy_"+ext+".pkl", 'rb') as run:
        scy = pickle.load(run)

def sim(lam, eps, die, divi, batch_size=1, random_state=None):

    ll = len(lam)

    if ll == 1:
        param = [[float(lam), float(eps), float(die), float(divi)]]
    else:
        for i in range(ll):
            if i == 0:
                param = [lam[i],eps[i],die[i],divi[i]]
            else:
                param = np.vstack((param,[lam[i],eps[i],die[i],divi[i]]))

    param = sc.transform(param)

    if ML == "NN":

        y = model.predict(param,use_multiprocessing=True,max_queue_size=100,workers=50)

        if len(y) == 1:
            y = np.concatenate(y)
            y = y[:5]
        elif len(y) == 5:
            y = y[:5]
        else:
            y = y[:,:5]

        y = scy.inverse_transform(y)

    elif ML =="GP":

        y = model.sample_y(param,1,random_state=None)

        y_D = []

        for i, j in enumerate(y):
            j = np.transpose(j)
            if len(y_D) < 1:
                y_D = j
            else:
                y_D = np.vstack((y_D,j))

        y = scy.inverse_transform(y_D)

    return y

print("OBS: ", y_obs)
print("STD: ", std)

def distance(y1,y2):
    return (y1["data"]-y2["data"])**2/std**2

lam = elfi.Prior('uniform', 0.01, 0.5)
eps = elfi.Prior('uniform', 0.01, 0.5)
die = elfi.Prior('uniform', 0.01, 0.5)
divi = elfi.Prior('uniform', 0.01, 0.5)

Y = elfi.Simulator(sim, lam, eps, die, divi, observed=y_obs)

def summer(x, i = 0):
    if len(x) == 1:
        x = np.concatenate(x)
        return x[i]/std[i]
    elif len(x) == 5:
        return x[i]/std[i]
    else:
        return x[:,i]/std[i]

S0 = elfi.Summary(summer, Y)
S1 = elfi.Summary(summer, Y, 1)
S2 = elfi.Summary(summer, Y, 2)
S3 = elfi.Summary(summer, Y, 3)
S4 = elfi.Summary(summer, Y, 4)

exists = glob(storage+"/Obs_*")

print(name)

if storage+"/Obs_"+name+".txt" in exists:
    print("Already exists")
else:
    d = elfi.Distance('euclidean', S0, S1, S2, S3, S4)

    if ML == "NN":
        rej = elfi.Rejection(d, batch_size=10000)
    elif ML == "GP":
        rej = elfi.Rejection(d, batch_size=1000)

    np.savetxt("Obs_"+name+".txt",np.column_stack((y_obs,std)))

    result = rej.sample(N, quantile=qs) # 0.0001

    epsilon = result.samples['eps']
    lambdaC = result.samples['lam']
    die = result.samples['die']
    div = result.samples['divi']

    samples1 = np.column_stack((epsilon,lambdaC,div,die))

    save_object(result, name+"_sam_emu.pkl")

    np.savetxt(name+"_sam_emu.txt",samples1)
