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

season = 'winter'
net, const_load_heatpump, const_load_household, time_steps = gd.setup_grid_irep(season)

#net, df_pv, df, pv_generators, const_load, const_pv = gd.setup_grid()
#time_steps = df_pv.index

print(net.load)

