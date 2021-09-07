import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

def readdir(direc, num):

    logfiles = glob(direc)

    numf = len(logfiles)

    print(numf)

    GSC = []
    GPP = []
    GDS = []
    Dead = []
    Rim = []

    for log in logfiles:
        data = np.genfromtxt(log)
        if data[-1,0] >= num:
            data = data[num,:]
            GSC.extend([data[2]])
            GPP.extend([data[3]])
            GDS.extend([data[4]])
            Dead.extend([data[5]])
            Rim.extend([data[6]])

    df = pd.DataFrame({'GSC': GSC,
               'GPP': GPP,
               'GDS': GDS,
               'Dead': Dead,
               'Rim': Rim,
                         })


    print(df.quantile([0.16,0.5,0.84]))
    print((df.quantile(0.84) - df.quantile(0.5))/np.sqrt(df.quantile(0.5)))

    return GSC, GPP, GDS, Dead, Rim, numf

direcs = glob("./E*")

for name in direcs:
    GSC, GPP, GDS, Dead, Rim, numf = readdir(name+"/run*/history.log", 99)
    results = np.column_stack((GSC,GPP,GDS,Dead,Rim))
    if numf == 250:
        np.savetxt("summary_"+name[2:]+".txt",results)
