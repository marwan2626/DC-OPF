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
net, const_load_heatpump, const_load_household, time_steps, df_season_heatpump_prognosis, df_heatpump, df_household = gd.setup_grid_irep(season)

Bbus = dt.calculate_bbus_matrix(net)

results = opf.solve_opf4(net, time_steps, const_load_heatpump, const_load_household, Bbus)
pl.plot_opf_results(results)

