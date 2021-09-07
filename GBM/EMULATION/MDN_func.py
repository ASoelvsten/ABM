from __future__ import absolute_import, division, print_function
import h5py
import pickle

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp

from tensorflow.keras.layers import Input, Dense, Activation, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# The code is based and inspired on the examples published by Oliver Borschers on https://colab.research.google.com/drive/1at5lIq0jYvA58AmJ0aVgn2aUVpzIbwS3

# Global parameters to be shared by functions

no_parameters = 3
no_mix = 3
dim_out = 5
components = no_mix*dim_out
neurons = 100

#=======================================================
# Function to load global parameters

def mdnglob():
    return no_mix, no_parameters, neurons, components, dim_out

#=======================================================
# Function for pickling:

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


#=======================================================
# Since the output of the MDN is the mean, the covariance matrix and the weights, it requires a few lines of code 
# to sample from the predicted distribution. This function provides an easy way to complish this.

def predicting(x,mdn,no_mix,dim_out,scy,no_samples=1000):
    pvec = mdn.predict(x)

    alpha = pvec[:,0:int(no_mix)]
    bs = len(alpha)
    mu = tf.reshape(pvec[:,int(no_mix):int(no_mix+no_mix*dim_out)], [bs,no_mix,dim_out])
    sigma = pvec[:,int(no_mix*dim_out+no_mix):int(no_mix*dim_out+no_mix+(no_mix*dim_out**2-no_mix*dim_out)*0.5+no_mix*dim_out)]

    cov = []
    for j in range(bs):
        Cs = []
        for i in range(no_mix):
            C = covmat(sigma[j],i,no_mix)
            Cs.extend([C])
        cov.extend([Cs])

    mmd = mixgauss_full(alpha,mu,cov)
    y_sample = ((tfd.Sample(mmd,sample_shape=no_samples)).sample()).numpy()
    y_sample = scy.inverse_transform(y_sample)

    return y_sample


#=======================================================
# MDN Class:

class MDN_Full(tf.keras.Model):

    def __init__(self, neurons=100, ncomp = 3, dim =5):
        super(MDN_Full, self).__init__(name="MDN_Full")
        """Initialize neural network that is to be used for predictions"""

        self.neurons = neurons
        self.ncomp = ncomp
        self.dim = dim

        self.h1 = Dense(neurons, activation="relu", name="h1")
        self.h2 = Dense(neurons, activation="relu", name="h2")
        self.h3 = Dense(neurons, activation="relu", name="h3")
        self.h4 = Dense(neurons, activation="relu", name="h4")

        self.alphas = Dense(ncomp, activation="softmax", name="alphas")
        self.mus = Dense(ncomp*dim, name="mus")
        num_sig = int((ncomp*dim**2-ncomp*dim)*0.5+ncomp*dim)
        self.sigmas = Dense(num_sig, activation="linear", name="sigmas") # nnelu is used below for sigma always to be positive. That is not needed anymore
        self.vec = Concatenate(name="vec")

    def call(self, inputs):
        """Specify neural network structure as well as the properties that this network will have to output"""

        x = self.h1(inputs) # Input layer
        x = self.h2(x) # Hidden Layer
        x = Dropout(rate = 0.1)(x) # Dropout layer
        x = self.h3(x) # Hidden layer
        x = Dropout(rate = 0.1)(x) # Dropout layer
        x = self.h4(x) # Hidden layer
        x = Dropout(rate = 0.1)(x) # Dropout layer

        alpha_v = self.alphas(x) # Output
        mu_v = self.mus(x) # Ouput
        sigma_v = self.sigmas(x) #Output

        return self.vec([alpha_v, mu_v, sigma_v])

#=======================================================
# Function that is able to slice the output of the neural network into the different components.

def slice_parameter_vectors_full(parameter_vector):
    alpha = parameter_vector[:,0:int(no_mix)]
    bs = alpha.get_shape().as_list()[0]
    mu = tf.reshape(parameter_vector[:,int(no_mix):int(no_mix+no_mix*dim_out)], [bs,no_mix,dim_out])
    sigma = parameter_vector[:,int(no_mix*dim_out+no_mix):int(no_mix*dim_out+no_mix+(no_mix*dim_out**2-no_mix*dim_out)*0.5+no_mix*dim_out)]

    cov = []
    for j in range(bs):
        Cs = []
        for i in range(no_mix):
            C = covmat(sigma[j],i,no_mix)
            Cs.extend([C])
        cov.extend([Cs])

    return alpha, mu, cov

#=======================================================
# Mixture model from tensor flow.

def mixgauss_full(alpha,mu,cov):

    mmd = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=alpha),
    components_distribution=tfd.MultivariateNormalFullCovariance(
        loc=mu,
        covariance_matrix=cov)
    )

    return mmd

#=======================================================
# Covariance matrix computation based on the paper by Alsing et al. (2019): Fast likelihood-free cosmology with neural density estimators and active learning.
# The way that we deal with the diagonal ensures e positive definiteness. 

def covmat(sigmas,i,ncomp):
    A = tf.cast(tfp.math.fill_triangular(sigmas[int(((dim_out**2-dim_out)*0.5+dim_out)*i):int(((dim_out**2-dim_out)*0.5+dim_out)*(i+1))]),dtype=np.float32)
    A = A - tf.linalg.diag(tf.linalg.diag_part(A)) + tf.linalg.diag(tf.exp(tf.linalg.diag_part(A)))
    A = tf.tensordot(A,tf.transpose(A),1)
    return A

#=======================================================
# Loss function. No assumed batch size.

def loss_func(y, parameter_vector):

    alpha_all, mu_all, cov_all = slice_parameter_vectors_full(parameter_vector) # Unpack parameter vectors

    mmd = mixgauss_full(alpha_all,mu_all,cov_all)

    log_likelihood = mmd.log_prob(y) # Evaluate log-probability of y

    ll_accuracy = -tf.reduce_mean(log_likelihood, axis=-1)

    return ll_accuracy

#=======================================================
# This function provides the loss function for for a batch of 1 simulation, while the above functions are able to deal with several simulations.
# You can you either depending on your preferences.

def loss_func_single(y, parameter_vector):

    alpha = parameter_vector[:,0:int(no_mix)][0]
    mu = tf.reshape(parameter_vector[:,int(no_mix):int(no_mix+no_mix*dim_out)], [no_mix,dim_out])
    sigmas = parameter_vector[:,int(no_mix*dim_out+no_mix):int(no_mix*dim_out+no_mix+(no_mix*dim_out**2-no_mix*dim_out)*0.5+no_mix*dim_out)][0]

    Cs = []
    for i in range(no_mix):
        A = tf.cast(tfp.math.fill_triangular(sigmas[int(((dim_out**2-dim_out)*0.5+dim_out)*i):int(((dim_out**2-dim_out)*0.5+dim_out)*(i+1))]),dtype=np.float32)
        A = A - tf.linalg.diag(tf.linalg.diag_part(A)) + tf.linalg.diag(tf.exp(tf.linalg.diag_part(A)))
        A = tf.tensordot(A,tf.transpose(A),1)
        Cs.extend([A])

    mmd = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=alpha),
    components_distribution=tfd.MultivariateNormalFullCovariance(
        loc=mu,
        covariance_matrix=Cs)
    )

    log_likelihood = mmd.log_prob(y) # Evaluate log-probability of y

    ll_accuracy = -tf.reduce_mean(log_likelihood, axis=-1)

    return ll_accuracy

#=======================================================
# Function for loading grid.

def load_grid(name):

    f = h5py.File(name, 'r')
    header = f['header'][:]
    X = f['grid'][:,1:5]
    y = f['grid'][:,[5,6,7,8,9]]

    header = header[1:]
    Head = []
    for i, h in enumerate(header):
        if i != len(header)-1:
#             Head.extend([h.decode("utf-8")])
             Head.extend([h])
        else:
#             Head.extend([h[:-2].decode("utf-8")])
             Head.extend([h[:-2]])

    print(Head)

    X_clean = []
    y_clean = []
    removed = 0

    datal = len(X)

    for i in range(datal):
        if y[i,0] == 0 or y[i,1] == 0: # or y[i,2] == 0 or y[i,3] == 0 or y[i,4] == 0:
            removed += 1
        elif len(X_clean) > 0:
            X_clean = np.vstack((X_clean,X[i,:]))
            y_clean = np.vstack((y_clean,y[i,:]))
        else:
            X_clean = X[i,:]
            y_clean = y[i,:]

    print("%i of %i simulations removed" % (removed, len(X)))

    X = X_clean
    y = y_clean

    print("Grid header:", Head)

    return X, y, Head
