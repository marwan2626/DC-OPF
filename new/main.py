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

#net, df_pv, df, pv_generators, const_load, const_pv  = gd.setup_grid()
season = 'winter'
net, ds_load_household, const_load_heatpump, ds_load_household, const_load_household = gd.setup_grid_irep(season)

Bbus = dt.calculate_bbus_matrix(net)

time_steps = ds_load_household.index

results = opf.solve_opf2(net, time_steps, const_load_heatpump, const_load_household, Bbus)
pl.plot_opf_results(results)

