import GBM

Restart = "None"
store = "RUN1"
resx = 201  # Number of cells x
resy = 201  # Number of cells y
steps = 350 # Time-steps
SavN = 2    # Save every N
Dox = 5.
lambdaC = 0.1
O2_crit = 0.1
low = 5
Dup = 200
trapped = 8
Qup = 100
epsilon = 0.2
dx = 1.
dy = dx
Pdie = 0.05
Pprol = 0.02
Pdivi = 0.25
myplots = True

GBM.run_ABM(Restart, store, resx, resy, steps, SavN, Dox, lambdaC, O2_crit, low, Dup, trapped, Qup, epsilon, Pdie, Pprol, Pdivi, dx, dy, myplots)
