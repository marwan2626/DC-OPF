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
import drcc as drcc
import montecarlo as mc

#### PACKAGES ####
import pandapower.plotting as pp
import matplotlib.pyplot as plt

season = 'winter'
net, const_load_household, const_load_heatpump, time_steps, df_season_heatpump_prognosis, df_household, df_heatpump, heatpump_scaling_factors = gd.setup_grid_irep(season)

Bbus = dt.calculate_bbus_matrix(net)
results = drcc.drcc_opf2(net, time_steps, const_load_heatpump, const_load_household, Bbus, df_season_heatpump_prognosis, df_heatpump, heatpump_scaling_factors,  max_iter_drcc=100, alpha=0.05, eta=5e-4)
#results = opf.solve_opf6(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors, Bbus)
pl.plot_opf_results_plotly(results,net)

#results = drcc.drcc_opf(net, time_steps, const_load_heatpump, const_load_household, Bbus, df_season_heatpump_prognosis, df_heatpump, max_iter_drcc=100, alpha=0.05, eta=1e-5)


#pl.plot_opf_results(results)

