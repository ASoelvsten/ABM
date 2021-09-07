import numpy as np
import glob, os
from progress.bar import Bar
import h5py

def distil_hdf5(loc, name):

    files = glob.glob(loc+"/*/varied.txt")

    table = []
    header = ["Time-step"]

    with Bar('Constructing...', max = len(files)) as bar:
        for j, fname in enumerate(files):
            logname = fname[:-10]+"history.log"
            if os.path.isfile(logname):
                params = []
                with open(fname) as f:
                    lines = f.readlines()
                    for line in lines:
                        words = line.split()
                        params.extend([float(words[1])])
                        if j == 0:
                            header.extend([words[0]])

                f.close()
                history = np.genfromtxt(logname)
                snapshots = len(history[:,0])
                if j == 0:
                    with open(logname) as f:
                        lines = f.readlines()
                        words = lines[1].split(",")
                        header.extend(list(words[2:]))    
                    f.close()

                for i in range(snapshots):
                    if history[i,0] == 99:
                        newrow = []
                        newrow.extend([history[i,0]])
                        newrow.extend(params)
                        newrow.extend(list(history[i,2:]))
                        if len(table) == 0:
                            table = newrow
                        else:
                            table = np.vstack((table, newrow))
            bar.next()

    asciiList = [n.encode("ascii", "ignore") for n in header]
    print(asciiList)
    hf = h5py.File(name,"w")
    hf.create_dataset("header", data=asciiList)
    hf.create_dataset("grid", data=table)
    hf.close()

loc = "./LIB/DIM4_10/"
name = "grid_0_dim4_100m.h5"

distil_hdf5(loc, name)
