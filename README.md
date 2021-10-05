# ABM
This repository contains the code for two stochastic agent-based models (ABM): one for tumour growth (GBM) and one for the spread of infectious diseases (SIRDS). The codes are compiled using cython. Each folder contains a bash script for recompiling the ABM. Moreover, each folder contains a python script called execute.py that shows how to import the compiled cython code as a function.

In addition, the repository contains scripts for constructing grids of each of the ABMs and tools for inferences. These tools cover neural networks (NN), Gaussian processes (GP) and mixture density models (MDN).

If you use our code in your research, please cite our paper *Efficient inference for agent-based models of real-world phenomena* on [https://www.biorxiv.org/content/10.1101/2021.10.04.462980v1](https://www.biorxiv.org/content/10.1101/2021.10.04.462980v1).
