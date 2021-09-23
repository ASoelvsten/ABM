import numpy as np
import pickle
import pygtc
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from glob import glob
from scipy.stats import wasserstein_distance as wd
import os.path
import tensorflow as tf
from MDN_func import mdnglob, loss_func, MDN_Full, save_object, load_grid, mixgauss_full
from MDN_func import predicting, loss_func_single
from tensorflow.keras.layers import Activation
from sklearn.model_selection import train_test_split
from tensorflow_probability import distributions as tfd

ext = "103_clean"
ML = "MDN" # NN, GP, or MDN

#=====================================================
# Create array with summary statistics of data and predictions

def summa(Results, GSC, i, samples):
    GSC_median = np.percentile(Results[:,i],50)
    GSC_std1 = np.percentile(Results[:,i],84) - np.percentile(Results[:,i],16)
    GSC_std2 = np.std(Results[:,i])
    GSC_std3 = np.percentile(Results[:,i],5)
    GSC_std4 = np.percentile(Results[:,i],95)
    GSC_mean = np.mean(Results[:,i])

    s_median = np.percentile(samples[:,i],50)
    sstd1 = np.percentile(samples[:,i],84)-np.percentile(samples[:,i],16)
    sstd2 = np.std(samples[:,i])
    sstd3 = np.percentile(samples[:,i],5)
    sstd4 = np.percentile(samples[:,i],95)
    s_mean = np.mean(samples[:,i])

    WD = wd(Results[:,i],samples[:,i])

    res_GSC = [s_median, s_mean, sstd1, sstd2, sstd3, sstd4, GSC_median, GSC_mean, GSC_std1, GSC_std2, GSC_std3, GSC_std4, WD]
    if len(GSC) < 1:
        GSC = res_GSC
    else:
        GSC = np.vstack((GSC,res_GSC))

    return GSC

#=====================================================

data = glob("../DIST/summary_*.txt")

if ML == "NN":

    model = tf.keras.models.load_model("emu_model_"+ext+".h5", compile=False)

    with open("emu_sc_"+ext+".pkl", 'rb') as run:
        sc = pickle.load(run)

    with open("emu_scy_"+ext+".pkl", 'rb') as run:
        scy = pickle.load(run)

if ML == "GP":

    with open("emu_GP_model_"+ext+".pkl", 'rb') as run:
        model = pickle.load(run)

    with open("emu_GP_sc_"+ext+".pkl", 'rb') as run:
        sc = pickle.load(run)

    with open("emu_GP_scy_"+ext+".pkl", 'rb') as run:
        scy = pickle.load(run)

if ML == "MDN":

    with open("emu_MDN2_sc_"+ext+".pkl", 'rb') as run:
        sc = pickle.load(run)

    with open("emu_MDN2_scy_"+ext+".pkl", 'rb') as run:
        scy = pickle.load(run)

    # To load weights ...
    no_mix, no_parameters, neurons, components, dim_out = mdnglob()

    opt = tf.keras.optimizers.Adam(learning_rate=1e-5,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-09,
          ) 

    folder = "../GRID/"

    if "104" in ext:
        grid_name = "grid_0_dim4_100.h5"
    if "103" in ext:
        grid_name = "grid_0_dim4_100l.h5"
    if "105" in ext:
        grid_name = "grid_0_dim4_100h.h5"
    if "102" in ext:
        grid_name = "grid_0_dim4_100vl.h5"

    X, y, header = load_grid(folder + grid_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
    x_train = sc.transform(X_train)
    x_val = sc.transform(X_val)
    y_train = scy.transform(y_train)
    y_val = scy.transform(y_val)

    model = MDN_Full(neurons=neurons, ncomp=no_mix,dim=dim_out)
    model.compile(loss=loss_func_single, optimizer=opt)
    model.fit(x=x_train, y=y_train, epochs=1, validation_data=(x_val, y_val), batch_size=1, verbose=1)
    model.load_weights("emu_MDN2_model_"+ext+".h5")

OBS = []
GSC = []
GPP = []
GDS = []
Dead = []
Rim = []

num = len(data)

for j, f in enumerate(data):

    Results = []

    eps = float(f[f.find("E_")+2:f.find("_L_")])
    lam = float(f[f.find("_L_")+3:f.find("_D_")])
    div = float(f[f.find("_D_")+3:f.find("_cd_")])
    die = float(f[f.find("_cd_")+4:f.find(".txt")])

    print(f)

    print("lam, eps, die, div: ", lam, eps, die, div)

    name = "E_"+str(eps)+"_L_"+str(lam)+"_D_"+str(div)+"_cd_"+str(die)

    if os.path.isfile("../DIST/summary_"+name+".txt"):

        param = [[lam, eps, die, div]]

        param = sc.transform(param)

        print("Running %i of %i..." % (j+1, num))

        samples = np.genfromtxt("../DIST/summary_"+name+".txt")

        CI = np.percentile(samples,84,axis=0)-np.percentile(samples,16,axis=0)

        if 0.0 not in CI:

            if len(OBS) < 1:
                OBS = [lam, eps, die, div]
            else:
                OBS = np.vstack((OBS, [lam, eps, die, div]))

            print("Based on %i samples" % (len(samples)))

            if ML == "NN":
                for i in range(len(samples)):
 
                    print("Running %f per cent" % ((100*float(i)+1.)/len(samples)), end="\r")

                    y = model.predict(param)

                    y = y[:,:5]

                    y = scy.inverse_transform(y)

                    y = np.asarray(np.concatenate(y))
 
                    if len(Results) > 0:
                         Results = np.vstack((Results,y))
                    else:
                        Results = y

            elif ML == "GP":

                y = model.sample_y(param,len(samples),random_state=None)

                y_D = np.transpose(y[0])

                Results = scy.inverse_transform(y_D)
            
            elif ML == "MDN":

                y_sample = predicting(param,model,no_mix,dim_out,scy,no_samples = len(samples))

                Results = y_sample[0]

            GSC = summa(Results, GSC, 0, samples)
            GPP = summa(Results,  GPP, 1, samples)
            GDS = summa(Results,  GDS, 2, samples)
            Dead = summa(Results,  Dead, 3, samples)
            Rim = summa(Results, Rim, 4, samples)

Res = np.column_stack((OBS,GSC,GPP,GDS,Dead,Rim))
np.savetxt("Res_emu_"+ext+"_"+ML+".txt", Res)
