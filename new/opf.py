"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

OPF File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

#### SCRIPTS ####
import data as dt

# Function to solve OPF problem over a time series
def solve_opf2(net, time_steps, const_pv, const_load, Bbus):
    model = gp.Model("opf_with_dc_load_flow")

    # Define the costs for import and export
    import_cost = 10  # €/MW for importing power from the external grid
    export_cost = 8  # €/MW for exporting power to the external grid
    curtailment_cost = 150  # €/MW for curtailing PV (set higher than import/export costs)

    epsilon = 100e-9  # Small positive value to ensure some external grid usage


    # Initialize decision variables
    pv_gen_vars = {}  # Store PV generation decision variables
    ext_grid_import_vars = {}  # Store external grid import power decision variables
    ext_grid_export_vars = {}  # Store external grid export power decision variables
    theta_vars = {}  # Store voltage angle decision variables (radians)
    curtailment_vars = {} # Store decision variables for curtailment

    slack_bus_index = net.ext_grid.bus.iloc[0]

    # Pre-process Bbus: Remove the row and column corresponding to the slack bus
    Bbus_reduced = np.delete(Bbus, slack_bus_index, axis=0)
    Bbus_reduced = np.delete(Bbus_reduced, slack_bus_index, axis=1)

    # Dictionaries to store results
    pv_gen_results = {}
    load_results = {}
    ext_grid_import_results = {}
    ext_grid_export_results = {}
    theta_results = {}
    line_results = {}

    # Add variables for each time step
    for t in time_steps:
        # Update const_pv and const_load for this time step
        const_pv.time_step(net, time=t)
        const_load.time_step(net, time=t)

        # Extract the bus indices where PV generators are connected (from net.sgen.bus)
        pv_buses = net.sgen.bus.values

        # Create PV generation variables for this time step
        pv_gen_vars[t] = model.addVars(pv_buses, lb=0, ub=net.sgen.p_mw.values, name=f'pv_gen_{t}')
        curtailment_vars[t] = model.addVars(pv_buses, lb=0, ub=net.sgen.p_mw.values, name=f'curtailment_{t}')
        for bus in pv_buses:
            # Find the index in sgen corresponding to this bus
            sgen_index = np.where(net.sgen.bus.values == bus)[0][0]
            model.addConstr(curtailment_vars[t][bus] == net.sgen.p_mw.values[sgen_index] - pv_gen_vars[t][bus], 
                            name=f'curtailment_constraint_{t}_{bus}')
            
        # External grid power variables for import and export at the slack bus (bus 0)
        ext_grid_import_vars[t] = model.addVar(lb=0, name=f'ext_grid_import_{t}')  # Import is non-negative
        ext_grid_export_vars[t] = model.addVar(lb=0, name=f'ext_grid_export_{t}')  # Export is non-negative
        model.addConstr(ext_grid_import_vars[t] + ext_grid_export_vars[t] >= epsilon, name=f'nonzero_ext_grid_usage_{t}')

        # Voltage angle variables for all buses
        theta_vars[t] = model.addVars(net.bus.index, lb=-GRB.INFINITY, name=f'theta_{t}')

        # Fix the slack bus angle to 0 radians
        model.addConstr(theta_vars[t][slack_bus_index] == 0, name=f'slack_theta_{t}')

        # Store loads for each bus at this time step
        load_results[t] = {bus: net.load.loc[net.load.bus == bus, 'p_mw'].values[0] if bus in net.load.bus.values else 0 for bus in net.bus.index}

    # Add power balance and load flow constraints for each time step
    for t in time_steps:
        # Power injection vector P
        P_injected = {bus: gp.LinExpr() for bus in net.bus.index}

        for i, bus in enumerate(net.bus.index):
            if bus in net.load.bus.values:
                # Subtract the load
                P_injected[bus] -= net.load.loc[net.load.bus == bus, 'p_mw'].values[0]

            if bus in pv_buses:
                # Only add PV generation if the bus has PV (i.e., in net.sgen.bus)
                P_injected[bus] += pv_gen_vars[t][bus]

            if bus == slack_bus_index:
                # Add the import minus export for the external grid power at the slack bus
                P_injected[bus] += ext_grid_import_vars[t] - ext_grid_export_vars[t]

        model.update()

        #for bus in net.bus.index:
            #print(f"Time step {t}, Bus {bus}: Power injected (MW) = {P_injected[bus]}")

        # Convert P_injected to per unit
        P_pu = {bus: P_injected[bus] / net.sn_mva for bus in net.bus.index}

        # Reduce the power vector by removing the slack bus entry
        P_pu_reduced = [P_pu[bus] for bus in net.bus.index if bus != slack_bus_index]

        # Power flow constraint: P_pu_reduced = Bbus_reduced * theta_reduced
        theta_reduced_vars = [theta_vars[t][i] for i in net.bus.index if i != slack_bus_index]

        # Apply power balance constraints for each non-slack bus
        for i in range(len(Bbus_reduced)):
            power_balance_expr = gp.LinExpr()
            for j in range(len(Bbus_reduced)):
                power_balance_expr += Bbus_reduced[i, j] * theta_reduced_vars[j]

            model.addConstr(P_pu_reduced[i] == power_balance_expr, name=f'power_flow_{t}_{i}')

        # Total power balance constraint at the slack bus
        # This enforces that the slack bus always balances generation and demand
        total_generation = gp.quicksum(pv_gen_vars[t][bus] for bus in pv_buses)
        total_load = gp.quicksum(net.load.loc[net.load.bus == bus, 'p_mw'].values[0] for bus in net.load.bus.values)
        model.addConstr(ext_grid_import_vars[t] - ext_grid_export_vars[t] == total_load - total_generation, name=f'power_balance_slack_{t}')


    # Line power flow and loading constraints (with the corrected expression)
    for t in time_steps:
        line_results[t] = {
            "line_pl_mw": {},
            "line_loading_percent": {},
            "line_current_mag": {}
        }

        for line in net.line.itertuples():
            from_bus = line.from_bus
            to_bus = line.to_bus
            base_voltage = net.bus.at[from_bus, 'vn_kv'] * 1e3  # Convert kV to V
            x_pu = line.x_ohm_per_km * line.length_km / ((base_voltage ** 2) / net.sn_mva)

            # Power flow on this line: (theta_from - theta_to) / X
            power_flow_expr = (theta_vars[t][from_bus] - theta_vars[t][to_bus]) / x_pu
            power_flow_mw = power_flow_expr * net.sn_mva / 1e6  # Convert to MW

            sqrt3 = np.sqrt(3)
            current_mag_ka = power_flow_mw / (sqrt3 * (base_voltage / 1e3))

            # Create an auxiliary variable for the absolute value of the current magnitude
            abs_current_mag_ka = model.addVar(lb=0, name=f'abs_current_mag_ka_{line.Index}_{t}')
            model.addConstr(abs_current_mag_ka >= current_mag_ka, name=f'abs_current_mag_ka_pos_{line.Index}_{t}')
            model.addConstr(abs_current_mag_ka >= -current_mag_ka, name=f'abs_current_mag_ka_neg_{line.Index}_{t}')

            # Now, calculate the line loading percentage using the auxiliary variable
            if hasattr(line, 'max_i_ka'):
                line_loading_percent = 100 * (abs_current_mag_ka / line.max_i_ka)
                model.addConstr(line_loading_percent <= 100, name=f'line_loading_{t}_{line.Index}')

            # Store results for each line in the time step
            line_results[t]["line_pl_mw"][line.Index] = power_flow_mw
            line_results[t]["line_loading_percent"][line.Index] = line_loading_percent            
            line_results[t]["line_current_mag"][line.Index] = current_mag_ka

    # Objective: Minimize total cost (import, export, and curtailment costs)
    total_cost = gp.quicksum(import_cost * ext_grid_import_vars[t] + 
                            export_cost * ext_grid_export_vars[t] + 
                            curtailment_cost * curtailment_vars[t][bus] 
                            for t in time_steps for bus in pv_buses)
    model.setObjective(total_cost, GRB.MINIMIZE)

    # After adding all constraints and variables
    model.setParam('Presolve', 0)
    model.update()

    # Optimize the model
    model.optimize()

    # Check if optimization was successful
    if model.status == gp.GRB.OPTIMAL:
        # Extract optimized values for PV generation, external grid power, loads, and theta
        for t in time_steps:
            pv_gen_results[t] = {bus: pv_gen_vars[t][bus].x for bus in pv_buses}
            ext_grid_import_results[t] = ext_grid_import_vars[t].x
            ext_grid_export_results[t] = ext_grid_export_vars[t].x
            theta_results[t] = {bus: theta_vars[t][bus].x for bus in net.bus.index}

            # Extract numerical values for line results
            for line in net.line.itertuples():
                line_results[t]["line_pl_mw"][line.Index] = line_results[t]["line_pl_mw"][line.Index].getValue()
                line_results[t]["line_loading_percent"][line.Index] = line_results[t]["line_loading_percent"][line.Index].getValue()
                line_results[t]["line_current_mag"][line.Index] = line_results[t]["line_current_mag"][line.Index].getValue()

            # After optimization, print the key variable results
            print(f"Time Step {t}:")
            print(f"PV Generation: {[pv_gen_vars[t][bus].x for bus in pv_buses]}")
            print(f"External Grid Import: {ext_grid_import_vars[t].x}")
            print(f"External Grid Export: {ext_grid_export_vars[t].x}")
            print(f"Theta (angles): {[theta_vars[t][bus].x for bus in net.bus.index]}")

            #for line in net.line.itertuples():
                #print(f"Line {line.Index}: Power Flow MW = {line_results[t]['line_pl_mw'][line.Index]}, Loading % = {line_results[t]['line_loading_percent'][line.Index]}")


        # Return results in a structured format
        results = {
            'pv_gen': pv_gen_results,
            'load': load_results,
            'ext_grid_import': ext_grid_import_results,
            'ext_grid_export': ext_grid_export_results,
            'theta': theta_results,  # Add theta results to the final results
            'line_results': line_results  # Line-specific results added
        }
        
        return results
    
    elif model.status == gp.GRB.INFEASIBLE:
        # If the model is infeasible, write the model to an ILP file for debugging
        print("Optimization failed - model is infeasible. Writing model to 'infeasible_model.ilp'")
        model.computeIIS()  # Compute IIS to identify the infeasible set
        model.write("infeasible_model.ilp")
        return None
    else:
        print(f"Optimization failed with status: {model.status}")
        return None

