"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

Main File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### SCRIPTS ####
import data as dt
import griddata as gd 
import results as rs
import plot as pl
import opf as opf

net, df_pv, df, pv_generators, const_load, const_pv  = gd.setup_grid()

print(net.load)
#Bbus = dt.calculate_bbus_matrix(net)

#time_steps = df_pv.index

#results = opf.solve_opf2(net, time_steps, const_pv, const_load, Bbus)
#pl.plot_opf_results(results)

