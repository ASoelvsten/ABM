#!/bin/bash
echo "Cythonize"
# If you want the html script
#cython -a SIRD.pyx
python3 setup2.py build_ext --inplace
echo "Start run"
python3 execute.py
