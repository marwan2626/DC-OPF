"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

DRCC-OPF File
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
import opf 


def calculate_covariance_matrix(heatpumpForecast):
    # Ensure 'sigma P' is present in the DataFrame
    if 'stdP' not in heatpumpForecast.columns:
        raise ValueError("DataFrame must contain 'stdP' for standard deviation values.")

    # Extract the 'stdP' values and square them to get the variance
    variance = (heatpumpForecast['stdP']/180000) ** 2

    # Create a diagonal covariance matrix from the variance
    covariance_matrix = np.diag(variance)

    return covariance_matrix

def calculate_sensitivity(heatpumpForecast, opf_results, opf_results_t, time_steps):
    # Extract baseline currents from the initial OPF results for timestep 0
    opf_currents_prev = extract_line_currents(opf_results)
    
    # Initialize a dictionary to store sensitivities for each line and timestep
    sensitivity_results = {line: {} for line in opf_currents_prev.keys()}

    # Loop over each timestep from 1 to calculate di/dw using the change from t-1 to t
    for t in range(1, len(time_steps)):  # Start from 1 to compare with previous timestep
        # Calculate the deviation (w) for timestep t and t-1
        w_t = heatpumpForecast['stdP'].iloc[t]    # Forecast deviation at timestep t
        w_prev = heatpumpForecast['stdP'].iloc[t - 1]  # Forecast deviation at timestep t-1

        # Use provided OPF results for current timestep and get currents
        opf_currents_t = extract_line_currents(opf_results_t[t])

        # Calculate di/dw for each line as the change in current divided by change in w
        for line in opf_currents_prev.keys():
            if w_t != w_prev:  # Avoid division by zero in case of no change
                sensitivity_results[line][time_steps[t]] = (opf_currents_t[line] - opf_currents_prev[line]) / (w_t - w_prev)
            else:
                sensitivity_results[line][time_steps[t]] = 0  # No change in sensitivity if w is constant

        # Update opf_currents_prev to the current timestep's results
        opf_currents_prev = opf_currents_t

    return sensitivity_results

def calculate_omega_I(alpha, sensitivity, cov_matrix):
    # Compute the product of Sensitivity and Covariance
    product = np.dot(sensitivity, cov_matrix)
    
    # Calculate the L2-norm of the product
    norm_L2 = np.linalg.norm(product, ord=2)
    
    # Calculate Omega_I using the specified formula
    omega_I = np.sqrt((1 - alpha) / alpha) * norm_L2
    return omega_I

def solve_opf(net, time_steps, const_load_heatpump, const_load_household, Bbus, Omega_I):
    model = gp.Model("opf_with_dc_load_flow")

    # Define the costs for import and export
    import_cost = 100  # €/MW for importing power from the external grid
    export_cost = 80  # €/MW for exporting power to the external grid
    curtailment_cost = 150  # €/MW for curtailing PV (set higher than import/export costs)
    flexibility_cost = 120  # €/MW for reducing load at bus 1

    epsilon = 100e-9  # Small positive value to ensure some external grid usage


    # Initialize decision variables
    pv_gen_vars = {}  # Store PV generation decision variables
    ext_grid_import_vars = {}  # Store external grid import power decision variables
    ext_grid_export_vars = {}  # Store external grid export power decision variables
    theta_vars = {}  # Store voltage angle decision variables (radians)
    curtailment_vars = {} # Store decision variables for curtailment
    flexible_load_vars = {}  # New flexible load variables
    flex_load_curtailment_vars = {}  # New flexible load curtailment variables


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

    # Temporary dictionary to store updated load values per time step
    time_synchronized_loads = {t: {} for t in time_steps}

    # Identify buses with flexible loads
    flexible_load_buses = net.load[net.load['controllable'] == True].bus.values
    print(f"Flexible load buses: {flexible_load_buses}")
    # Add variables for each time step
    for t in time_steps:
        # Update const_pv and const_load for this time step
        const_load_heatpump.time_step(net, time=t)
        const_load_household.time_step(net, time=t)

        # Extract the time-synchronized load for each bus
        # Ensure all buses have an entry in `time_synchronized_loads` for the given time step
        for bus in net.bus.index:
            if bus in net.load.bus.values:
                # If the bus has a load, store its value
                time_synchronized_loads[t][bus] = net.load.loc[net.load.bus == bus, 'p_mw'].values[0]
            else:
                # For buses without loads, set to zero or a default value
                time_synchronized_loads[t][bus] = 0.0

        # Extract the bus indices where PV generators are connected (from net.sgen.bus)
        pv_buses = net.sgen.bus.values

        # Create PV generation variables for this time step
        if len(pv_buses) > 0:
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

        # Flexible load variable at bus 1   
        flexible_load_vars[t] = model.addVars(
            flexible_load_buses,
            lb=0,
            ub={bus: time_synchronized_loads[t][bus] for bus in flexible_load_buses},
            name=f'flexible_load_{t}'
        )
            
        flex_load_curtailment_vars[t] = model.addVars(
        flexible_load_buses,
        lb=0,
        ub={bus: time_synchronized_loads[t][bus] for bus in flexible_load_buses},
        name=f'flex_load_curtailment_{t}'
        )

        for bus in flexible_load_buses:
            original_load = time_synchronized_loads[t][bus]
            model.addConstr(flex_load_curtailment_vars[t][bus] == original_load - flexible_load_vars[t][bus],
                            name=f'flex_load_curtailment_constraint_{t}_{bus}')

    # Add power balance and load flow constraints for each time step
    for t in time_steps:
        # Power injection vector P
        P_injected = {bus: gp.LinExpr() for bus in net.bus.index}

        for i, bus in enumerate(net.bus.index):
            if bus in net.load.bus.values:
                if bus in flexible_load_buses:
                    # Use the flexible load variable for controllable loads
                    P_injected[bus] -= flexible_load_vars[t][bus]
                else:
                    # For non-flexible loads, use the time-synchronized load
                    P_injected[bus] -= time_synchronized_loads[t][bus]

            if len(pv_buses) > 0 and bus in pv_buses:
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
        total_generation = gp.quicksum(pv_gen_vars[t][bus] for bus in pv_buses) if pv_buses.size > 0 else 0
        total_load = gp.quicksum(flexible_load_vars[t][bus] for bus in flexible_load_buses) + gp.quicksum(net.load.loc[net.load.bus == bus, 'p_mw'].values[0] 
                                for bus in net.load.bus.values if bus not in flexible_load_buses)        
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
                #
                for line in net.line.itertuples():
                    from_bus = line.from_bus
                    to_bus = line.to_bus

                    # Retrieve the timestep-specific, line-specific Omega_I
                    Omega_I_timestep_line = Omega_I.get((line.Index, t), 0)

                    # Apply the tightening term to the line loading upper bound
                    line_loading_percent_ub = 100 - Omega_I_timestep_line
                    line_loading_percent = 100 * (abs_current_mag_ka / line.max_i_ka)
                    model.addConstr(line_loading_percent <= line_loading_percent_ub, name=f'line_loading_{t}_{line.Index}')

            # Store results for each line in the time step
            line_results[t]["line_pl_mw"][line.Index] = power_flow_mw
            line_results[t]["line_loading_percent"][line.Index] = line_loading_percent            
            line_results[t]["line_current_mag"][line.Index] = current_mag_ka

    # Objective: Minimize total cost (import, export, and curtailment costs)
    total_cost = gp.quicksum(import_cost * ext_grid_import_vars[t] +
                         export_cost * ext_grid_export_vars[t] +
                         (gp.quicksum(curtailment_cost * curtailment_vars[t][bus] for bus in pv_buses) if len(pv_buses) > 0 else 0) +
                         gp.quicksum(flexibility_cost * flex_load_curtailment_vars[t][bus] for bus in flexible_load_buses)
                         for t in time_steps)
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
            
            load_results[t] = {
                bus: flexible_load_vars[t][bus].x if bus in flexible_load_buses else 
                time_synchronized_loads[t][bus]
                for bus in net.bus.index
            }

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


    # Compute the product of Sensitivity and Covariance
    product = np.dot(sensitivity, cov_matrix)
    
    # Calculate the L2-norm of the product
    norm_L2 = np.linalg.norm(product, ord=2)
    
    # Calculate Omega_I using the specified formula
    omega_I = np.sqrt((1 - alpha) / alpha) * norm_L2
    return omega_I

def drcc_opf(net, time_steps, const_load_heatpump, const_load_household, Bbus, heatpumpForecast, max_iter_drcc=10, max_iter_opf=5, alpha=0.05, tol=1e-4, eta=1e-3):
    Omega_I = 0.0
    prev_Omega_I = None

    # Initial covariance calculation (run once, as per your earlier step)
    cov_matrix = calculate_covariance_matrix(heatpumpForecast)

    for drcc_iter in range(max_iter_drcc):
        # Adjust omega based on covariance and latest sensitivity values
        opf_results = solve_opf(net, time_steps, const_load_heatpump, const_load_household, Bbus, Omega_I)

        if opf_results is None:
            print(f"OPF infeasible in DRCC iteration {drcc_iter + 1}")
            continue
        
        # Calculate the latest sensitivity using the baseline results
        sensitivity = calculate_sensitivity(heatpumpForecast, opf_results, opf_results_t, time_steps)
        
        # Calculate Omega_I for constraint tightening
        omega_I = calculate_omega_I(alpha, sensitivity, cov_matrix)

        # Pass omega_I to solve_opf3 to update the line loading constraints
        opf_results = solve_opf(net, time_steps, const_load_heatpump, const_load_household, Bbus, omega_I)

        # Check convergence criteria based on omega changes
        if prev_Omega_I is not None and np.max(np.abs(Omega_I - prev_Omega_I)) < eta:
            print(f"Converged in {drcc_iter + 1} DRCC iterations.")
            break
        prev_Omega_I = Omega_I


