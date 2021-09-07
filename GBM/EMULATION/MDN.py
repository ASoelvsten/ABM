# This script contains code for training the mixture density network. All the underlying code can be found in MDN_func.py.

from __future__ import absolute_import, division, print_function
import h5py
import pickle
from sklearn.metrics import r2_score
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from MDN_func import mdnglob, save_object, load_grid
from MDN_func import loss_func, MDN_Full, slice_parameter_vectors_full, mixgauss_full
from MDN_func import predicting
from MDN_func import loss_func_single

# To choose different resolutions of the training data, reset ext (to 102_clean, 103_clean, 104_clean or 105_clean, where e.g. 104_clean corresponds to the grid with 10^4 simulations).

ext = "103_clean"

no_mix, no_parameters, neurons, components, dim_out = mdnglob()

opt = tf.keras.optimizers.Adam(learning_rate=1e-5,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-09,
      )

eager = False

# You can reset the patience based on the convergence of the neural network of the MDN.

mdn = MDN_Full(neurons=neurons, ncomp=no_mix,dim=dim_out)
if eager:
    mdn.compile(loss=loss_func, optimizer=opt, run_eagerly=True)
    mon = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=50, verbose=0, mode='auto')
else:
    mdn.compile(loss=loss_func_single, optimizer=opt)
    mon = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=50, verbose=0, mode='auto')

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

sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test) 
x_val = sc.transform(X_val)
scy = StandardScaler()
y_train = scy.fit_transform(y_train)
y_test = scy.transform(y_test)
y_val = scy.transform(y_val)

if eager:
    history = mdn.fit(x=x_train, y=y_train, epochs=5000, validation_data=(x_val, y_val),
                  batch_size=max([10,int(len(x_train)/50)]), verbose=1, shuffle=True,
#                  batch_size=1, verbose=1, shuffle=True,
                  use_multiprocessing=False, callbacks=[mon])
else:
    history = mdn.fit(x=x_train, y=y_train, epochs=5000, validation_data=(x_val, y_val),
                  batch_size=1, verbose=1, shuffle=True,
                  use_multiprocessing=False, callbacks=[mon])

# All the plots below are for the evaluation of the MDN. Take a close look at them before deciding whether or not the peformance of the MDN is acceptable. Do you over- or underfit?

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.savefig("history"+ext+".png")

y_sample = predicting(x_val,mdn,no_mix,dim_out,scy)
Y = np.mean(y_sample,axis=1)
plt.figure()
plt.plot(scy.inverse_transform(y_val)[:,0],Y[:,0],"mo",alpha=0.3)
plt.figure()
plt.plot(scy.inverse_transform(y_val)[:,1],Y[:,1],"ro",alpha=0.3)
plt.figure()
plt.plot(scy.inverse_transform(y_val)[:,2],Y[:,2],"yo",alpha=0.3)
plt.figure()
plt.plot(scy.inverse_transform(y_val)[:,3],Y[:,3],"ko",alpha=0.3)
plt.figure()
plt.plot(scy.inverse_transform(y_val)[:,4],Y[:,4],"go",alpha=0.3)
print(r2_score(scy.inverse_transform(y_val), Y))

y_sample = predicting(x_test,mdn,no_mix,dim_out,scy)
Y = np.mean(y_sample,axis=1)

plt.figure()
plt.plot(scy.inverse_transform(y_test)[:,0],Y[:,0],"mo",alpha=0.3)
plt.figure()
plt.plot(scy.inverse_transform(y_test)[:,1],Y[:,1],"ro",alpha=0.3)
plt.figure()
plt.plot(scy.inverse_transform(y_test)[:,2],Y[:,2],"yo",alpha=0.3)
plt.figure()
plt.plot(scy.inverse_transform(y_test)[:,3],Y[:,3],"ko",alpha=0.3)
plt.figure()
plt.plot(scy.inverse_transform(y_test)[:,4],Y[:,4],"go",alpha=0.3)
print(r2_score(scy.inverse_transform(y_test), Y))

x_pred = sc.transform([[0.1,0.2,0.05,0.25]])

y_sample = predicting(x_pred,mdn,no_mix,dim_out,scy)

print(np.mean(y_sample[0],axis=0))

plt.figure()
plt.hist(y_sample[0][:,0],linestyle = "--" ,bins =20, lw =2, color= "m", histtype=u'step')
plt.hist(y_sample[0][:,1],linestyle = "--" ,bins =20, lw =2, color= "r", histtype=u'step')
plt.hist(y_sample[0][:,2],linestyle = "--" ,bins =20, lw =2, color= "y", histtype=u'step')
plt.hist(y_sample[0][:,3],linestyle = "--" ,bins =20, lw =2, color= "k", histtype=u'step')
plt.hist(y_sample[0][:,4],linestyle = "--" ,bins =20, lw =2, color= "g", histtype=u'step')

plt.show()

# Here, we save all the output to h5 and pickled files.

mdn.save_weights("emu_MDN2_model_"+ext+".h5")
save_object(sc, "emu_MDN2_sc_"+ext+".pkl")
save_object(scy, "emu_MDN2_scy_"+ext+".pkl")

save_object(x_test, "emu_MDN2_xtest_"+ext+".pkl")
save_object(y_test, "emu_MDN2_ytest_"+ext+".pkl")
save_object(x_val, "emu_MDN2_xval_"+ext+".pkl")
save_object(y_val, "emu_MDN2_yval_"+ext+".pkl")
save_object(x_train, "emu_MDN2_xtrain_"+ext+".pkl")
save_object(y_train, "emu_MDN2_ytrain_"+ext+".pkl")

