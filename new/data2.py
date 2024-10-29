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
import pandapower.plotting as pp
import matplotlib.pyplot as plt

season = 'winter'
net, const_load_heatpump, const_load_household, time_steps, df_season_heatpump = gd.setup_grid_irep(season)

Bbus = dt.calculate_bbus_matrix(net)

results = drcc.drcc_opf(net, time_steps, const_load_heatpump, const_load_household, Bbus, df_season_heatpump, max_iter_drcc=10, alpha=0.05, eta=1e-3)

pl.plot_opf_results(results)







#net, df_pv, df, pv_generators, const_load, const_pv = gd.setup_grid()
#time_steps = df_pv.index
#print(net.load)
""" 
def print_const_control_info(const_control):
    print(f"ConstControl info for {const_control}")
    print(f"Element type: {const_control.element}")
    print(f"Controlled element indices: {const_control.element_index}")
    print(f"Column: {const_control.variable}")
    print(f"Time series profile data:\n{const_control.profile_name}")


print_const_control_info(const_load_heatpump) """

#plot.simple_plot(net)

#
""" # Define load and non-load buses
load_buses = net.load.bus.unique()
non_load_buses = net.bus.index.difference(load_buses)

# Create a plot with geodata using matplotlib
fig, ax = plt.subplots(figsize=(12, 10))

# Plot lines with geodata
line_collection = pp.create_line_collection(net, color="gray", linewidths=1)
pp.draw_collections([line_collection], ax=ax)

# Plot non-load buses in blue
non_load_bus_collection = pp.create_bus_collection(net, buses=non_load_buses, color="blue", size=0.1)
pp.draw_collections([non_load_bus_collection], ax=ax)

# Plot load buses in red
load_bus_collection = pp.create_bus_collection(net, buses=load_buses, color="red", size=0.1)
pp.draw_collections([load_bus_collection], ax=ax)

# Add bus indices as text labels
for idx, (x, y) in net.bus_geodata[['x', 'y']].iterrows():
    ax.text(x, y, str(idx), fontsize=10, ha='center', va='center', color='black')

# Set title and show plot
ax.set_title("Network Plot with Load Buses Highlighted")
plt.show() """