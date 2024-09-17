"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

Plot File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####

import matplotlib.pyplot as plt

def plot_load_p_mw(load_p_mw):
    plt.figure(figsize=(10, 5))
    for column in load_p_mw.columns:
        plt.plot(load_p_mw.index, load_p_mw[column], label=f'Load {column}')
    plt.xlabel('Time Step')
    plt.ylabel('Load Power (MW)')
    plt.title('Load Power over Time')
    plt.legend()
    plt.show()

def plot_sgen_p_mw(sgen_p_mw):
    plt.figure(figsize=(10, 5))
    for column in sgen_p_mw.columns:
        plt.plot(sgen_p_mw.index, sgen_p_mw[column], label=f'Load {column}')
    plt.xlabel('Time Step')
    plt.ylabel('PV Power (MW)')
    plt.title('PV generation over Time')
    plt.legend()
    plt.show()

def plot_line_loading_percent(line_loading_percent):
    plt.figure(figsize=(10, 5))
    for column in line_loading_percent.columns:
        plt.plot(line_loading_percent.index, line_loading_percent[column], label=f'Load {column}')
    plt.xlabel('Time Step')
    plt.ylabel('Loading Percentage %')
    plt.title('Line Loading over Time')
    plt.legend()
    plt.show()


def plot_line_loading_percent2(loading_percent):
    plt.figure(figsize=(10, 5))
    time_steps = loading_percent.index  # Use the index of loading_percent as the time steps
    for column in loading_percent.columns:
        plt.plot(time_steps, loading_percent[column], label=f'Line {column}')
    plt.xlabel('Time Step')
    plt.ylabel('Loading Percentage %')
    plt.title('Line Loading over Time')
    plt.legend()
    plt.show()

# Function to plot line current magnitude
def plot_line_current_magnitude(i_ka):
    plt.figure(figsize=(10, 5))
    time_steps = i_ka.index  # Use the index of i_ka as the time steps
    for column in i_ka.columns:
        plt.plot(time_steps, i_ka[column], label=f'Line {column}')
    plt.xlabel('Time Step')
    plt.ylabel('Current Magnitude (kA)')
    plt.title('Line Current Magnitude over Time')
    plt.legend()
    plt.show()

# Function to plot line power flow
def plot_line_power_flow(pl_mw):
    plt.figure(figsize=(10, 5))
    time_steps = pl_mw.index  # Use the index of pl_mw as the time steps
    for column in pl_mw.columns:
        plt.plot(time_steps, pl_mw[column], label=f'Line {column}')
    plt.xlabel('Time Step')
    plt.ylabel('Power Flow (MW)')
    plt.title('Line Power Flow over Time')
    plt.legend()
    plt.show()

def plot_opf_results(results):
    # Extract the results from the dictionary
    pv_gen = results['pv_gen']
    load = results['load']
    ext_grid_import = results['ext_grid_import']
    ext_grid_export = results['ext_grid_export']
    theta = results['theta']
    line_results = results['line_results']
    
    # Get the list of time steps
    time_steps = list(pv_gen.keys())

    # Plot PV Generation for each bus
    plt.figure(figsize=(10, 6))
    for bus in pv_gen[time_steps[0]].keys():
        pv_values = [pv_gen[t][bus] for t in time_steps]
        plt.plot(time_steps, pv_values, label=f'Bus {bus}')
    plt.xlabel('Time Steps')
    plt.ylabel('PV Generation (MW)')
    plt.title('PV Generation by Bus and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Load for each bus
    plt.figure(figsize=(10, 6))
    for bus in load[time_steps[0]].keys():
        load_values = [load[t][bus] for t in time_steps]
        plt.plot(time_steps, load_values, label=f'Bus {bus}')
    plt.xlabel('Time Steps')
    plt.ylabel('Load (MW)')
    plt.title('Load by Bus and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot External Grid Import/Export
    plt.figure(figsize=(10, 6))
    ext_import_values = [ext_grid_import[t] for t in time_steps]
    ext_export_values = [ext_grid_export[t] for t in time_steps]
    plt.plot(time_steps, ext_import_values, label='External Grid Import (MW)', color='green')
    plt.plot(time_steps, ext_export_values, label='External Grid Export (MW)', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Power (MW)')
    plt.title('External Grid Import and Export over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Theta (Voltage Angles) for each bus
    plt.figure(figsize=(10, 6))
    for bus in theta[time_steps[0]].keys():
        theta_values = [theta[t][bus] for t in time_steps]
        plt.plot(time_steps, theta_values, label=f'Bus {bus}')
    plt.xlabel('Time Steps')
    plt.ylabel('Voltage Angle Theta (Radians)')
    plt.title('Voltage Angle Theta by Bus and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Line Power Flow (MW) for each line
    plt.figure(figsize=(10, 6))
    for line in line_results[time_steps[0]]['line_pl_mw'].keys():
        line_pl_mw_values = [line_results[t]['line_pl_mw'][line] for t in time_steps]
        plt.plot(time_steps, line_pl_mw_values, label=f'Line {line}')
    plt.xlabel('Time Steps')
    plt.ylabel('Line Power Flow (MW)')
    plt.title('Line Power Flow (MW) by Line and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Line Loading Percentage for each line
    plt.figure(figsize=(10, 6))
    for line in line_results[time_steps[0]]['line_loading_percent'].keys():
        line_loading_values = [line_results[t]['line_loading_percent'][line] for t in time_steps]
        plt.plot(time_steps, line_loading_values, label=f'Line {line}')
    plt.xlabel('Time Steps')
    plt.ylabel('Line Loading (%)')
    plt.title('Line Loading Percentage by Line and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Line Current Magnitude (kA) for each line
    plt.figure(figsize=(10, 6))
    for line in line_results[time_steps[0]]['line_current_mag'].keys():
        line_current_values = [line_results[t]['line_current_mag'][line] for t in time_steps]
        plt.plot(time_steps, line_current_values, label=f'Line {line}')
    plt.xlabel('Time Steps')
    plt.ylabel('Line Current Magnitude (kA)')
    plt.title('Line Current Magnitude (kA) by Line and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

