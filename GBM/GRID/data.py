import numpy as np
from glob import glob

parfiles = glob("./LIB/GBM*/param*")

P = []

for pf in parfiles:
    data = np.genfromtxt(pf)

    lam = data[5]
    eps = data[11]
    die = data[12]
    div = data[14]
    param = [lam,eps,die,div]
    if len(P) < 1:
        P = param
    else:
        P = np.vstack((P, param))

np.savetxt("Examples.txt", P)
