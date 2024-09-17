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
#print(net.switch)
Bbus = dt.calculate_bbus_matrix(net)
#print(Bbus)
time_steps = df_pv.index
#time_steps = list(range(75)) 

#results = dt.manual_dc_timeseries(time_steps, net, const_pv, const_load, Bbus)
# Load results into separate dataframes
#theta_degrees, loading_percent, load_p_mw, sgen_p_mw, line_pl_mw, i_ka = rs.load_results(results)
# Plot line loading percentage over time
#pl.plot_line_loading_percent(loading_percent)
# Plot line current magnitude over time
#pl.plot_line_current_magnitude(i_ka)
#print(net.bus)
#print(net.line)
#print(net.switch)
#print(net.trafo)
#print(net.load)
#print(net.sgen)



results = opf.solve_opf2(net, time_steps, const_pv, const_load, Bbus)
pl.plot_opf_results(results)

""" results_opf = opf.solve_opf_direct_load_flow(net, pv_generators, time_steps, df_pv, const_pv, const_load, Bbus)
#Load results into separate dataframes
loading_percent, ext_grid_p_mw, sgen_p_mw, curtailment_pv_mw, load_p_mw = rs.load_results_opf(results_opf)
# Plot line loading percentage over time
pl.plot_line_loading_percent(loading_percent)
# Plot line current magnitude over time
#pl.plot_line_current_magnitude(i_ka)
pl.plot_sgen_p_mw(sgen_p_mw)
# Plot line power flow over time
pl.plot_line_power_flow(load_p_mw) """

""" #print(Ybus)
#print(Bbus)
#print(net.bus)
#print(net.line)
#print(net.switch)
#print(net.trafo)
#print(net.load)
#print(net.sgen)
time_steps = df_pv.index
results = dt.manual_dc_timeseries(time_steps, net, const_pv, const_load, Bbus)


# Load results into separate dataframes
theta_degrees, loading_percent, load_p_mw, sgen_p_mw, line_pl_mw, i_ka = rs.load_results(results)

#print("Loading Percent Columns:", loading_percent.columns)
#print("Line Power Flow Columns:", line_pl_mw.columns)
#print("Line Current Magnitude Columns:", i_ka.columns)

# Plot line loading percentage over time
pl.plot_line_loading_percent(loading_percent)

# Plot line current magnitude over time
pl.plot_line_current_magnitude(i_ka)

# Plot line power flow over time
pl.plot_line_power_flow(line_pl_mw)

pl.plot_sgen_p_mw(sgen_p_mw) """