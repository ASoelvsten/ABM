import SIRDS

Population = 12100
Pdie = 0.1
Pinf = 0.2
REN = 5.1
immune = 5000.
tinf = 7.1
store = "./SIR1"

#SIRDS.SIRABM(store, 0.1, 3, 0.184, 1225, Pdie, Pinf, REN, tinf, immune, 5000, False)
SIRDS.SIRABM(store, 0.1, 3, 0.17356, Population, Pdie, Pinf, REN, tinf, immune, 5000, False)
#SIRDS.SIRABM(store, 0.1, 3, 0.184, Population, Pdie, Pinf, REN, tinf, immune, 5000, True)
