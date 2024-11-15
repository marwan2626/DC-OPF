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


season = 'winter'
net, const_load_heatpump, const_load_household, time_steps, df_season_heatpump_prognosis, df_heatpump, df_household = gd.setup_grid_irep(season)

Bbus = dt.calculate_bbus_matrix(net)

results = drcc.drcc_opf(net, time_steps, const_load_heatpump, const_load_household, Bbus, df_season_heatpump_prognosis, df_heatpump, max_iter_drcc=100, alpha=0.05, eta=1e-5)

# curtailment = rs.curtailment_calculation(results, df_heatpump)

# pl.plot_curtailment(curtailment, time_steps)

#results = opf.solve_opf4(net, time_steps, const_load_heatpump, const_load_household, Bbus)

pl.plot_opf_results(results)

#all_results = mc.montecarlo_analysis_parallel(net, time_steps, df_season_heatpump_prognosis, df_household, n_jobs=-1)

#l.plot_line_current_histogram(all_results, net, line_index=0, time_step=437)

# # Plotting 'stdP' and 'meanP'
# plt.figure(figsize=(12, 6))
# plt.plot(df_season_heatpump['stdP'], label='stdP', color='b')
# plt.plot(df_season_heatpump['meanP'], label='meanP', color='r')
# plt.xlabel("Time Step")
# plt.ylabel("Power (arbitrary units)")
# plt.title("Heat Pump Forecast: Standard Deviation and Mean of Power")
# plt.legend()
# plt.grid(True)
# plt.show()





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