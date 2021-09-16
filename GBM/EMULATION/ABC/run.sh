#!/bin/bash
parallel --version

vals=($(seq 0 1 250))

parallel python3 rej.py {1} ::: "${vals[@]}"
