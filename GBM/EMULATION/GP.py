import h5py
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pygtc
import pickle
import h5py
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, RationalQuadratic
from sklearn.metrics import r2_score

# To train on another dataset, change ext. E.g. 103_clean_RBF refers to the grid with 10^3 data points.
# RBF refers to the kernel used.

ext = "103_clean_RBF"

#=======================================================
# Function for pickling:

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#=======================================================
# Function for loading grid

def load_grid(name):
    f = h5py.File(name, 'r')
    header = f['header'][:]
    X = f['grid'][:,1:5]
    y = f['grid'][:,[5,6,7,8,9]]

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

#=======================================================
# Function for cleaning grid. For instance, tumours, in which all cells are dead are sorted out.

def clean(X,y, header, newgrid):

    print("Removing dead tumours...")

    Dead_tumours = []
    X_clean = []
    y_clean = []
    removed = 0

    datal = len(X)

    for i in range(datal):
        if y[i,0] == 0 or y[i,1] == 0:
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
    y = y_clean

    return X, y, header

#=======================================================
# Main function. Training and testing the random forest or neural network.

def create_ML(X,y,header, dim = 4):

    print("Splitting test and training data...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print("Scaling...")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    scy = StandardScaler()
    y_train = scy.fit_transform(y_train)
    y_test = scy.transform(y_test)

    print("Fitting...")

    kernel = ConstantKernel(1.0, constant_value_bounds=(1.e-6,1e5)) * RBF(1.0, length_scale_bounds=(1.0e-2,1.e2)) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-3,0.5))
    regressor = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5, normalize_y=False, copy_X_train=True, random_state=None)

    regressor.fit(X_train, y_train)
    print(regressor.kernel_)

    y_pred = scy.inverse_transform(regressor.predict(X_test))

    y_test = scy.inverse_transform(y_test)

    print("Testing...")

    save_object(regressor, "emu_GP_model_"+ext+".pkl")

    save_object(sc, "emu_GP_sc_"+ext+".pkl")

    save_object(scy, "emu_GP_scy_"+ext+".pkl")

    for i in range(dim):
        print(header[i])
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test[:,i], y_pred[:,i]))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test[:,i], y_pred[:,i]))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test[:,i], y_pred[:,i])))
        plt.figure(i)
        plt.plot(y_test[:,i],y_pred[:,i],'o',alpha=0.4, label=header[i])
        plt.legend()
        print("Estimated variance "+header[i]+": " , 1 - np.var(y_test[:,i]-y_pred[:,i])/np.var(y_test[:,i]))
        plt.figure(dim+1)
        y = abs(y_pred[:,i] - y_test[:,i])/np.max(abs(y_pred[:,i] - y_test[:,i]))
        plt.plot(y_test[:,i]/np.max(y_test[:,i]),y,'o',label=header[i])
        plt.ylabel(r'$|y-\hat(y)|/max|y-\hat(y)|$')

    plt.show()

    print(r2_score(y_pred, y_test))

    return regressor, sc, scy

#====================================================
# Calls to run

def run_fit(grid_name, folder):

    X, y, header = load_grid(folder + grid_name)

    X, y, header = clean(X,y, header, "clean_" + grid_name)

    regressor, sc, scy = create_ML(X, y, header)

    return regressor, sc, scy

#====================================================

if "104" in ext:
    grid_name = "grid_0_dim4_100.h5"
if "103" in ext:
    grid_name = "grid_0_dim4_100l.h5"
if "105" in ext:
    grid_name = "grid_0_dim4_100h.h5"
if "102" in ext:
    grid_name = "grid_0_dim4_100vl.h5"

folder = "../GRID/"

regressor, sc, scy = run_fit(grid_name, folder)
