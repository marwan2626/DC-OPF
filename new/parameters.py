"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

Parameter File
"""

###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####
import gurobipy as gp
import numpy as np

###############################################################################
## GUROBI PARAMETERS ## 
###############################################################################
# gp.setParam("NonConvex",-1) # enable non convex constraints, enable = 2
gp.setParam("OutputFlag",0) # solver output, enable = 1
# gp.setParam("DualReductions", 0) # check if feasible or unbounded: enable = 0
# gp.setParam("MIPGap",2e-4) # MIP gap, default = 1e-4


###############################################################################
## GENERAL ## 
###############################################################################
### NETWORK ###


### TIME HORIZON ###

    
### CONVERGENCE CRITERIA ###
ETA_LF = 1e-4 # bfs standalone #ETA_BFS RENAMED
ETA_OPF = 5e-4 # bfs-opf voltage mismatch #ETA_BFSOPF RENAMED
#ETA_MARG_V = 1e-1 # bus voltage uncertainty margin


### ITERATION COUNTERS ###
M_MAX = 1 # maximum iterations outer CC loop
M_MIN = 1 # minimum iterations for outer CC loop
B_MAX = 5 # maximum iterations opf
K_MAX = 5 # maximum inner lf iterations


### FORECAST ###
#V_FCST = 1 # forecast version, for definition see forecast script header
#PV_MAX = 8 # 8 kWp installations for data set to normalize
N_DAY = 2 # number of days for monte-carlo simulation


###############################################################################
## FLAGS: DISABLE = 0 , ENABLE = 1 ## 
###############################################################################
### UNITS ###
#FLGBAT = 1 # BESS
#FLGSHED = 0 # load shedding
#FLGSHIFT = 0 # load shifting
#FLGCURT = 1 # active power curtailment
#FLGOLTC = 0 # OLTC trafo
#FLGLOAD = 1 # load profile: 0 = constant, 1 = time varying
#FLGPF = 0 # power factor limit PV inverters
#FLGPV = 0 # installed capacity PV from input file: 0 = input data, 1 = load dependent
#FLGCC = 0 # chance constraints
#FLGDRCC = 0 # distributionally robust or gaussian: 1 = DR, 0 = Gaussian



###############################################################################
## CHANCE-CONSTRAINTS ## 
###############################################################################
### UNCERTAINTY MARGIN ###
# power ratio gamma
#FLGCC_GMA = 0 # pre-defined gamma or from OPF: pre-defined = 0 - from OPF = 1
#power_factor = 0.95 # pre-defined power factor
#CC_GMA = np.sqrt((1-power_factor**2)/power_factor**2) # pre-defined power ratio


