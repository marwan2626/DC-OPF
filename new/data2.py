"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

Data File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####
import pandapower as pp
import pandas as pd
import numpy as np
from math import sqrt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#### SCRIPTS ####
import griddata as gd


def run_dc_load_flow(Bbus, net, P):
    # Identify the slack bus (usually the bus connected to ext_grid)
    slack_bus_index = net.ext_grid.bus.iloc[0]
    
    # Determine active buses (excluding the slack bus)
    active_buses = np.setdiff1d(np.arange(len(net.bus)), [slack_bus_index])
    
    # Reduce the Bbus matrix by removing the row and column corresponding to the slack bus
    Bbus_reduced = np.delete(np.delete(Bbus, slack_bus_index, axis=0), slack_bus_index, axis=1)
    
    # Solve for voltage angles for the reduced system (excluding slack bus)
    theta = np.linalg.solve(Bbus_reduced, P[active_buses])
    
    # Initialize theta_full to include all buses
    theta_full = np.zeros(len(net.bus), dtype=np.float64)
    
    # Insert the calculated angles for active buses (excluding slack bus)
    theta_full[active_buses] = theta

    # Initialize empty lists to store line results with correct line indices
    line_pl_mw = []
    line_current_mag = []
    line_loading_percent = []

    print("Line Index | From Bus | To Bus | Calculated Power Flow (MW) | Current (A)")

    # Calculate the power flow on each line and store results in the correct order
    for idx in net.line.index:
        from_bus = net.line.at[idx, 'from_bus']
        to_bus = net.line.at[idx, 'to_bus']
        X_line = net.line.at[idx, 'x_ohm_per_km'] * net.line.at[idx, 'length_km']  # Only consider reactance
        
        # Power flow calculation based on voltage angles difference
        power_flow = (theta_full[from_bus] - theta_full[to_bus]) / X_line
        
        # Calculate results
        pl_mw = power_flow * net.bus.at[from_bus, 'vn_kv']  # Power flow in MW
        current_mag = np.abs(power_flow)  # Current magnitude in per unit
        loading_percent = 100 * np.abs(current_mag) / net.line.at[idx, 'max_i_ka']
        
        # Store the results using the original line index
        line_pl_mw.append((idx, pl_mw))
        line_current_mag.append((idx, current_mag))
        line_loading_percent.append((idx, loading_percent))
        
        # Debugging print statement
        print(f"{idx:<10} | {from_bus:<8} | {to_bus:<7} | {pl_mw:<28.6f} | {current_mag:<12.6f}")

    # Convert lists to Pandas Series and ensure the results are indexed correctly
    results = {
        'theta_degrees': pd.Series(np.degrees(theta_full), index=net.bus.index),
        'line_pl_mw': pd.Series(dict(line_pl_mw), index=net.line.index),  # Power flow in MW
        'line_current_mag': pd.Series(dict(line_current_mag), index=net.line.index),
        'line_loading_percent': pd.Series(dict(line_loading_percent), index=net.line.index)
    }
    
    return results



def manual_dc_timeseries(time_steps, net, const_pv, const_load, Bbus, bus_index_map):
    results = {
        "time_step": [],
        "theta_degrees": [],
        "line_loading_percent": [],
        "line_current_mag": [],
        "load_p_mw": [],
        "sgen_p_mw": [],
        "line_pl_mw": []
    }

    line_indices = None  # Placeholder for line indices

    for t in time_steps:
        # Update controls using const_pv and const_load
        const_pv.time_step(net, time=t)
        const_load.time_step(net, time=t)

        # Recalculate the power injection vector P immediately after the update
        P = np.zeros(len(net.bus), dtype=np.float64)
        if not net.load.empty:
            P[net.load.bus.values.astype(int)] -= net.load.p_mw.values.astype(np.float64)
        if not net.sgen.empty:
            P[net.sgen.bus.values.astype(int)] += net.sgen.p_mw.values.astype(np.float64)

        # Run the DC load flow calculation
        flow_results = run_dc_load_flow(Bbus, net, P, bus_index_map)

        # Capture the line indices from the first result
        if line_indices is None:
            line_indices = flow_results['line_pl_mw'].index

        # Log results with correct indexing
        results["time_step"].append(t)
        results["theta_degrees"].append(flow_results['theta_degrees'])
        results["line_loading_percent"].append(flow_results['line_loading_percent'].tolist())
        results["line_current_mag"].append(flow_results['line_current_mag'].tolist())
        results["load_p_mw"].append(net.load.p_mw.values.tolist())
        results["sgen_p_mw"].append(net.sgen.p_mw.values.tolist())
        results["line_pl_mw"].append(flow_results['line_pl_mw'].tolist())

    # Convert results to DataFrames, ensuring correct indices
    theta_degrees_df = pd.DataFrame(results["theta_degrees"], index=results["time_step"], columns=net.bus.index)
    line_loading_percent_df = pd.DataFrame(results["line_loading_percent"], index=results["time_step"], columns=line_indices)
    line_current_mag_df = pd.DataFrame(results["line_current_mag"], index=results["time_step"], columns=line_indices)
    load_p_mw_df = pd.DataFrame(results["load_p_mw"], index=results["time_step"], columns=net.load.index)
    sgen_p_mw_df = pd.DataFrame(results["sgen_p_mw"], index=results["time_step"], columns=net.sgen.index)
    line_pl_mw_df = pd.DataFrame(results["line_pl_mw"], index=results["time_step"], columns=line_indices)

    # Compile all results into a single DataFrame
    results_df = pd.concat({
        "theta_degrees": theta_degrees_df,
        "line_loading_percent": line_loading_percent_df,
        "line_current_mag": line_current_mag_df,
        "load_p_mw": load_p_mw_df,
        "sgen_p_mw": sgen_p_mw_df,
        "line_pl_mw": line_pl_mw_df
    }, axis=1)

    # Save results to an Excel file
    results_df.to_excel("output_results.xlsx")
    
    return results_df


def calculate_bbus_matrix(net):
    # Create a mapping from original bus indices to a sequential range
    bus_index_map = {bus_idx: i for i, bus_idx in enumerate(net.bus.index)}
    
    num_buses = len(net.bus)
    Bbus = np.zeros((num_buses, num_buses))  # Bbus is a real-valued matrix
    
    # Add line reactances
    for line in net.line.itertuples():
        from_bus = bus_index_map[line.from_bus]
        to_bus = bus_index_map[line.to_bus]
        x = line.x_ohm_per_km * line.length_km  # Consider only the reactance for Bbus
        
        # Bbus off-diagonal elements
        Bbus[from_bus, to_bus] -= 1 / x
        Bbus[to_bus, from_bus] -= 1 / x
        
        # Bbus diagonal elements
        Bbus[from_bus, from_bus] += 1 / x
        Bbus[to_bus, to_bus] += 1 / x
    
    # Add transformer reactances
    for trafo in net.trafo.itertuples():
        hv_bus = bus_index_map[trafo.hv_bus]
        lv_bus = bus_index_map[trafo.lv_bus]
        V_base_lv = trafo.vn_lv_kv
        x = (trafo.vk_percent / 100) * ( V_base_lv ** 2) / trafo.sn_mva  # Corrected reactance
    
        # Bbus off-diagonal elements
        Bbus[hv_bus, lv_bus] -= 1 / x
        Bbus[lv_bus, hv_bus] -= 1 / x
    
        # Bbus diagonal elements
        Bbus[hv_bus, hv_bus] += 1 / x
        Bbus[lv_bus, lv_bus] += 1 / x
    
    # Handle open switches by setting corresponding entries to zero
    for switch in net.switch.itertuples():
        if not switch.closed:
            from_bus = bus_index_map.get(switch.bus)
            to_bus = bus_index_map.get(switch.element)
            if from_bus is not None and to_bus is not None:
                Bbus[from_bus, to_bus] = 0
                Bbus[to_bus, from_bus] = 0
    
    return Bbus




