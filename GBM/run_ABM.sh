#!/bin/bash
echo "Cythonize"
cp ABM.pyx GBM.pyx
# If you want the html script
#cython -a GBM.pyx
python3 setup.py build_ext --inplace
rm GBM.pyx
echo "Start run"
python3 execute.py
