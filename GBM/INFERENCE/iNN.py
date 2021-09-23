# Script for training and validating neural networks for emulation.                                
# Note that only a few changes are necessary to change the script to an emulator.

from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
import numpy as np
from sklearn.metrics import r2_score
import tensorflow.keras
import elfi
import pygtc
import pickle
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

# To train on another dataset, change ext. E.g. 103_clean refers to the grid with 10^3 data points.
# We impose early stopping manually by adjusting the number of epochs (ne).

ext = "104_clean"
ne = 5000

#=======================================================================

# aleatoric loss function
# "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" by Y. Gal and Z. Ghahramani.
# "Evaluating Scalable Uncertainty Estimation Methods for DNN-Based Molecular Property Prediction" by G. Scalia et al.
# "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" Kendall & Gal
# Implementation is furthermore based on the code example by Michel Kana's blog "Uncertainty in Deep learning. How to measure?" to be found on 
# https://towardsdatascience.com/my-deep-learning-model-says-sorry-i-dont-know-the-answer-that-s-absolutely-ok-50ffa562cb0b

def aleatoric_loss(y_true, y_pred):
    se = K.pow((y_true[:,:4]-y_pred[:,:4]),2)
    inv_std = K.exp(-y_pred[:,4:])
    mse = K.mean(K.batch_dot(inv_std,se))
    reg = K.mean(y_pred[:,4:])
    return 0.5*(mse + reg)

#=======================================================================

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#=======================================================================

def load_grid(name):
    f = h5py.File(name, 'r')
    header = f['header'][:]
    y = f['grid'][:,1:5]
    X = f['grid'][:,[5,6,7,8,9]]

    header = header[1:]
    Head = []
    for i, h in enumerate(header):
        if i != len(header)-1:
#            Head.extend([h.decode("utf-8")])
            Head.extend([h]) #.decode("utf-8")])
        else:
#            Head.extend([h[:-2].decode("utf-8")])
            Head.extend([h[:-2]]) #.decode("utf-8")])

    print("Grid header:", Head)

    return X, y, Head

#=======================================================================

def clean(X,y, header):

    print("Removing dead tumours...")

    Dead_tumours = []
    X_clean = []
    y_clean = []
    removed = 0

    datal = len(X)

    for i in range(datal):
        if X[i,0] == 0 or X[i,1] == 0: # or X[i,2] == 0 or X[i,3] == 0 or X[i,4] == 0:
            Dead_tumours.extend([i])
            removed += 1
        elif len(X_clean) > 0:
            X_clean = np.vstack((X_clean,X[i,:]))
            y_clean = np.vstack((y_clean,y[i,:]))
        else:
            X_clean = X[i,:]
            y_clean = y[i,:]

    print("%i of %i simulations removed" % (removed, len(X)))

    X = X_clean
    if "log" in ext:
        X = np.log10(X)
    y = y_clean

    return X, y, header

#=======================================================================

def infer(folder, grid_name, training,neurons = 100,layers = 3,dropout_rate = 0.2,epochs = 5000):

    X, y, header = load_grid(folder + grid_name)

    X, y, header = clean(X,y, header,)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_val = sc.transform(X_val)

    scy = StandardScaler()
    y_train = scy.fit_transform(y_train)
    y_test = scy.transform(y_test)
    y_val = scy.transform(y_val)

    nr = np.zeros(len(y_train))
    y_train = np.column_stack((y_train,nr, nr, nr, nr))
    nr = np.zeros(len(y_val))
    y_val = np.column_stack((y_val,nr,nr,nr,nr))

    inputs = Input(shape=(5,))
    hl = Dense(100, kernel_initializer='uniform', activation='relu')(inputs)
    for i in range(layers):
        hl = Dense(neurons, kernel_initializer='uniform', activation='relu')(hl)
        hl = Dropout(rate = dropout_rate)(hl, training=training)
    outputs = Dense(8, kernel_initializer='uniform')(hl)
    model = Model(inputs, outputs)

    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-09,
            )
    model.compile(loss=aleatoric_loss, optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    history = model.fit(X_train, y_train, batch_size=int(len(X_train)/3), epochs = epochs, shuffle=True, validation_data=(X_val, y_val), use_multiprocessing=True)

    train_mse = model.evaluate(X_train, y_train, verbose=0)
    test_mse = model.evaluate(X_val, y_val, verbose=0)
    # plot loss during training
    plt.figure(1)
    plt.title('Loss / Mean Squared Error')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()

    y_pred =  scy.inverse_transform(model.predict(X_val)[:,:4])

    y_val =  scy.inverse_transform(y_val[:,:4])

    print(header)
    h = [0,1,2,3]

    for i in range(4):
        plt.figure(i+2)
        plt.plot(y_val[:,i],y_val[:,i],'r.')
        plt.plot(y_val[:,i],y_pred[:,i],'ko',alpha=0.4, label=header[h[i]])
        plt.figure(7)
        y = abs(y_pred[:,i] - y_val[:,i])/np.max(abs(y_pred[:,i] - y_val[:,i]))
        plt.plot(y_val[:,i]/np.max(y_val[:,i]),y,'o')

    print(r2_score(y_val, y_pred[:,:4]))

    model.save("infe_model_"+ext+".h5")
    save_object(sc, "infe_sc_"+ext+".pkl")
    save_object(scy, "infe_scy_"+ext+".pkl")

    save_object(X_test, "X_test_"+ext+".pkl")
    save_object(y_test, "y_test_"+ext+".pkl")

    plt.show()

    return model, sc, scy

#=======================================================================
# Function for testing with remaining unused data.

def testing():

    model = tensorflow.keras.models.load_model("infe_model_"+ext+".h5", compile=False)

    with open("infe_sc_"+ext+".pkl", 'rb') as run:
        sc = pickle.load(run)

    with open("infe_scy_"+ext+".pkl", 'rb') as run:
        scy = pickle.load(run)

    with open("X_test_"+ext+".pkl", 'rb') as f:
         X_test = pickle.load(f)

    with open("y_test_"+ext+".pkl", 'rb') as f:
         y_test = pickle.load(f)

    y_pred = scy.inverse_transform(model.predict(X_test)[:,:4])

    y_test = scy.inverse_transform(y_test[:,:4])

    for i in range(4):
        plt.figure(i+2)
        plt.plot(y_test[:,i],y_test[:,i],'r.')
        plt.plot(y_test[:,i],y_pred[:,i],'ko',alpha=0.4)
        plt.figure(7)
        y = abs(y_pred[:,i] - y_test[:,i])/np.max(abs(y_pred[:,i] - y_test[:,i]))
        plt.plot(y_test[:,i]/np.max(y_test[:,i]),y,'o')

    print(r2_score(y_test, y_pred[:,:5]))

    plt.show()

#=======================================================================

if "104" in ext:
    grid_name = "grid_0_dim4_100.h5"
if "103" in ext:
    grid_name = "grid_0_dim4_100l.h5"
if "105" in ext:
    grid_name = "grid_0_dim4_100h.h5"
if "102" in ext:
    grid_name = "grid_0_dim4_100vl.h5"

folder = "../GRID/"
training = True

model, sc, scy = infer(folder, grid_name, training,epochs=ne)
testing()
