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
# Function to solve OPF problem over a time series
def solve_opf(net, pv_generators, time_steps, df_pv, const_pv, const_load, Bbus):
    model = gp.Model("opf_day")
    power_cost = 100  # €/MW for both external grid power and losses

    # Define Variables for PV generation with bounds based on const_pv
    sgen_p_mw = model.addVars(
        time_steps, len(pv_generators),
        lb=0,  # Lower bound is always 0
        ub={(t, i): const_pv.data_source.get_time_step_value(t, 'pvgen') for t in time_steps for i in range(len(pv_generators))},
        name="sgen_p_mw"
    )

    # Define variable for external grid power (import/export) at each time step
    ext_grid_p_mw = model.addVars(time_steps, lb=-GRB.INFINITY, name="ext_grid_p_mw")

    # Auxiliary variable for the absolute value of external grid power
    ext_grid_abs_p_mw = model.addVars(time_steps, lb=0, name="ext_grid_abs_p_mw")

    # Add constraints to enforce that ext_grid_abs_p_mw[t] is the absolute value of ext_grid_p_mw[t]
    for t in time_steps:
        model.addConstr(ext_grid_abs_p_mw[t] >= ext_grid_p_mw[t], name=f"abs_ext_grid_pos_{t}")
        model.addConstr(ext_grid_abs_p_mw[t] >= -ext_grid_p_mw[t], name=f"abs_ext_grid_neg_{t}")

    # Objective: Minimize the total cost for both importing and exporting power
    total_cost = gp.quicksum(power_cost * ext_grid_abs_p_mw[t] for t in time_steps)
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Debug information storage
    debug_info = []

    # Constraints for Power Balance and Line Loading
    for t in time_steps:
        # Update PV generation limits
        for i, pv_gen in enumerate(pv_generators):
            max_p = df_pv.loc[t, 'pvgen']
            net.sgen.at[pv_gen, 'max_p_mw'] = max_p

        # Update time-series data
        const_pv.time_step(net, time=t)
        const_load.time_step(net, time=t)

        # Power balance constraint
        total_load_p_mw = net.load['p_mw'].sum()
        model.addConstr(
            gp.quicksum(sgen_p_mw[t, i] for i in range(len(pv_generators))) + ext_grid_p_mw[t] == total_load_p_mw,
            name=f"power_balance_{t}"
        )

        # Recalculate power injection vector and run DC load flow
        P = np.zeros(len(net.bus), dtype=np.float64)
        if not net.load.empty:
            P[net.load.bus.values.astype(int)] -= net.load.p_mw.values.astype(np.float64)
        if not net.sgen.empty:
            P[net.sgen.bus.values.astype(int)] += net.sgen.p_mw.values.astype(np.float64)

        flow_results = dt.run_dc_load_flow(Bbus, net, P)
        external_grid_power = flow_results['transformer_pl_mw'].iloc[0]

        # Debugging print for power balance and external grid power
        debug_info.append({
            'time_step': t,
            'total_load': total_load_p_mw,
            'ext_grid_power': external_grid_power,
            'line_loading': flow_results['line_loading_percent']
        })

    # Initialize line indices
    line_indices = None

    for t in time_steps:
        # Run the DC load flow calculation
        flow_results = dt.run_dc_load_flow(Bbus, net, P)

        # Ensure line_indices are set correctly based on the first time step
        if line_indices is None:
            line_indices = flow_results['line_pl_mw'].index

        # Add line loading constraints for each line based on the correct indices
        for idx, loading_percent in flow_results['line_loading_percent'].items():
            if pd.isna(loading_percent) or loading_percent < 0:
                print(f"Skipping invalid line loading constraint for line {idx}: {loading_percent}")
            else:
                # Use the correct line indices for each line loading constraint
                model.addConstr(loading_percent <= 100, name=f"line_loading_{line_indices[idx]}_{t}")

    # Solve the optimization model
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("infeasible_model.ilp")
        return None

    # Apply optimized values if the solution is optimal
    if model.status == GRB.OPTIMAL:
        for t in time_steps:
            for i, gen in enumerate(pv_generators):
                net.sgen.at[gen, 'p_mw'] = sgen_p_mw[t, i].X
            net.ext_grid.at[0, 'p_mw'] = ext_grid_p_mw[t].X

            P = np.zeros(len(net.bus), dtype=np.float64)
            if not net.load.empty:
                P[net.load.bus.values.astype(int)] -= net.load.p_mw.values.astype(np.float64)
            if not net.sgen.empty:
                P[net.sgen.bus.values.astype(int)] += net.sgen.p_mw.values.astype(np.float64)
            flow_results = dt.run_dc_load_flow(Bbus, net, P)

    # Print debug info for each time step
    for info in debug_info:
        print(f"Time step {info['time_step']}: Total Load = {info['total_load']}, External Grid Power = {info['ext_grid_power']}, Line Loading = {info['line_loading']}")

  # Step 7: Collect and return the results
    results = {
        "time_step": [],
        "line_loading_percent": [],
        "ext_grid_p_mw": [],
        "sgen_p_mw": [],
        "curtailment_pv_mw": [],
        "load_p_mw": []  # Add load power to results
    }

    for t in time_steps:
        # Calculate curtailment: max possible generation - actual generation
        curtailment_pv = [
            df_pv.loc[t, 'pvgen'] - sgen_p_mw[t, i].X for i in range(len(pv_generators))
        ]

        results["time_step"].append(t)
        results["line_loading_percent"].append(flow_results['line_loading_percent'].tolist())
        results["ext_grid_p_mw"].append(ext_grid_p_mw[t].X)
        results["sgen_p_mw"].append([sgen_p_mw[t, i].X for i in range(len(pv_generators))])
        results["curtailment_pv_mw"].append(curtailment_pv)
        results["load_p_mw"].append(net.load['p_mw'].tolist())  # Save load power

    # Step 8: Save the results in the same structure as your manual timeseries
    line_loading_percent_df = pd.DataFrame(results["line_loading_percent"], index=results["time_step"], columns=line_indices)
    ext_grid_p_mw_df = pd.DataFrame(results["ext_grid_p_mw"], index=results["time_step"], columns=[0])
    sgen_p_mw_df = pd.DataFrame(results["sgen_p_mw"], index=results["time_step"], columns=net.sgen.index)
    curtailment_pv_df = pd.DataFrame(results["curtailment_pv_mw"], index=results["time_step"], columns=net.sgen.index)
    load_p_mw_df = pd.DataFrame(results["load_p_mw"], index=results["time_step"], columns=net.load.index)

    # Combine all into a single DataFrame
    results_opf_df = pd.concat({
        "line_loading_percent": line_loading_percent_df,
        "ext_grid_p_mw": ext_grid_p_mw_df,
        "sgen_p_mw": sgen_p_mw_df,
        "curtailment_pv_mw": curtailment_pv_df,
        "load_p_mw": load_p_mw_df
    }, axis=1)

    # Save to Excel
    results_opf_df.to_excel("opf_results.xlsx")

    return results_opf_df


def solve_opf_direct_load_flow(net, pv_generators, time_steps, df_pv, const_pv, const_load, Bbus):
    model = gp.Model("opf_with_loadflow")
    power_cost = 100  # €/MW for both external grid power and losses

    # Define the optimization variables for PV generation with bounds based on sgen (provided by const_pv)
    pv_gen = model.addVars(
        time_steps, len(pv_generators),
        lb=0,  # Lower bound is 0
        ub={(t, i): net.sgen.at[pv_gen, 'p_mw'] for t in time_steps for i, pv_gen in enumerate(pv_generators)},  # Upper bound is the sgen value
        name="pv_gen"
    )

    # Define the optimization variable for external grid power (import/export) at each time step
    ext_grid_p_mw = model.addVars(time_steps, lb=-GRB.INFINITY, name="ext_grid_p_mw")

    # Auxiliary variable for the absolute value of external grid power
    ext_grid_abs_p_mw = model.addVars(time_steps, lb=0, name="ext_grid_abs_p_mw")

    # Absolute value of external grid power constraints
    for t in time_steps:
        model.addConstr(ext_grid_abs_p_mw[t] >= ext_grid_p_mw[t], name=f"abs_ext_grid_pos_{t}")
        model.addConstr(ext_grid_abs_p_mw[t] >= -ext_grid_p_mw[t], name=f"abs_ext_grid_neg_{t}")

    # Objective: Minimize the total cost for both importing and exporting power
    total_cost = gp.quicksum(power_cost * ext_grid_abs_p_mw[t] for t in time_steps)
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Constraints for power balance and line loading
    for t in time_steps:
        # Update load time series (uncontrollable)
        const_load.time_step(net, time=t)

        # Power balance constraint at each bus
        total_load_p_mw = net.load['p_mw'].sum()
        model.addConstr(
            gp.quicksum(pv_gen[t, i] for i in range(len(pv_generators))) + ext_grid_p_mw[t] == total_load_p_mw,
            name=f"power_balance_{t}"
        )

    # Solve the optimization model
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("infeasible_model.ilp")
        return None

    # Initialize results storage
    results = {
        "time_step": [],
        "line_loading_percent": [],
        "ext_grid_p_mw": [],
        "pv_gen": [],
        "curtailment_pv_mw": [],
        "load_p_mw": []  # Add load power to results
    }

    # Initialize line indices
    line_indices = None

    # After optimization, retrieve the optimized values for `pv_gen` and run the DC load flow calculation
    for t in time_steps:
        # Extract optimized `pv_gen` values for the current time step
        pv_gen_values = np.array([pv_gen[t, i].X for i in range(len(pv_generators))])

        # Construct the power injection vector P after optimization
        P = np.zeros(len(net.bus), dtype=np.float64)
        if not net.load.empty:
            P[net.load.bus.values.astype(int)] -= net.load.p_mw.values.astype(np.float64)
        if not net.sgen.empty:
            P[net.sgen.bus.values.astype(int)] += pv_gen_values

        # Perform load flow calculations
        flow_results = dt.run_dc_load_flow(Bbus, net, P)

        if line_indices is None:
            line_indices = flow_results['line_pl_mw'].index

        # Calculate curtailment: max possible generation (sgen) - actual generation (pv_gen)
        curtailment_pv = [net.sgen.at[pv_gen, 'p_mw'] - pv_gen_values[i] for i in range(len(pv_generators))]

        # Store results for the current time step
        results["time_step"].append(t)
        results["line_loading_percent"].append(flow_results['line_loading_percent'].tolist())
        results["ext_grid_p_mw"].append(ext_grid_p_mw[t].X)
        results["pv_gen"].append(pv_gen_values.tolist())
        results["curtailment_pv_mw"].append(curtailment_pv)
        results["load_p_mw"].append(net.load['p_mw'].tolist())

    # Step 8: Save the results in the same structure as your manual timeseries
    line_loading_percent_df = pd.DataFrame(results["line_loading_percent"], index=results["time_step"], columns=line_indices)
    ext_grid_p_mw_df = pd.DataFrame(results["ext_grid_p_mw"], index=results["time_step"], columns=[0])
    pv_gen_df = pd.DataFrame(results["pv_gen"], index=results["time_step"], columns=net.sgen.index)
    curtailment_pv_df = pd.DataFrame(results["curtailment_pv_mw"], index=results["time_step"], columns=net.sgen.index)
    load_p_mw_df = pd.DataFrame(results["load_p_mw"], index=results["time_step"], columns=net.load.index)

    # Combine all into a single DataFrame
    results_opf_df = pd.concat({
        "line_loading_percent": line_loading_percent_df,
        "ext_grid_p_mw": ext_grid_p_mw_df,
        "pv_gen": pv_gen_df,
        "curtailment_pv_mw": curtailment_pv_df,
        "load_p_mw": load_p_mw_df
    }, axis=1)

    # Save to Excel
    results_opf_df.to_excel("opf_results.xlsx")

    return results_opf_df


import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_opf2(net, time_steps, const_pv, const_load, Bbus):
    model = gp.Model("opf_with_dc_load_flow")

    # Define the costs for import and export
    import_cost = 100  # €/MW for importing power from the external grid
    export_cost = 100  # €/MW for exporting power to the external grid

    # Initialize decision variables
    pv_gen_vars = {}  # Store PV generation decision variables
    ext_grid_import_vars = {}  # Store external grid import power decision variables
    ext_grid_export_vars = {}  # Store external grid export power decision variables
    theta_vars = {}  # Store voltage angle decision variables (radians)

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

        # External grid power variables for import and export at the slack bus (bus 0)
        ext_grid_import_vars[t] = model.addVar(lb=0, name=f'ext_grid_import_{t}')  # Import is non-negative
        ext_grid_export_vars[t] = model.addVar(lb=0, name=f'ext_grid_export_{t}')  # Export is non-negative

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

        for bus in net.bus.index:
            print(f"Time step {t}, Bus {bus}: Power injected (MW) = {P_injected[bus]}")

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

    # Objective: Minimize the total cost of power import and export across all time steps
    total_cost = gp.quicksum(import_cost * ext_grid_import_vars[t] + export_cost * ext_grid_export_vars[t] for t in time_steps)
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

