"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

Monecarlo Analysis File
"""

###############################################################################
## IMPORT PACKAGES & SCRIPTS ##
###############################################################################
#### PACKAGES ####
import time
import numpy as np
import pandas as pd
import pandapower as pp
from joblib import Parallel, delayed
from tqdm import tqdm

#### SCRIPTS ####
import parameters as par

###############################################################################
## Generate Samples ##
###############################################################################

def generate_samples(df_season_heatpump_prognosis):
    # Convert meanP and stdP to float after replacing commas    
    n_samples = par.N_MC
    mc_samples = []
    
    for i in range(n_samples):
        # Create a copy of the original DataFrame structure, excluding stdP
        df_sample = df_season_heatpump_prognosis[['Unnamed: 0', 'meanP']].copy()
        
        # Generate samples for meanP using stdP as the scale (std deviation)
        df_sample['meanP'] = np.random.normal(loc=df_sample['meanP'], scale=df_season_heatpump_prognosis['stdP'])

        # Ensure that the meanP values are scaled correctly
        df_sample['P_HEATPUMP_NORM'] = df_sample['meanP'] / df_sample['meanP'].max()
        
        # Append to the list
        mc_samples.append(df_sample)
    
    # Print the header of the first sample for inspection
    print("Header of first sample:\n", mc_samples[0].head())
    
    return mc_samples

###############################################################################
## PANDAPOWER FUNCTIONS ##
###############################################################################


###############################################################################
## MONTECARLO ANALYSIS ##
###############################################################################

def run_single_sample(net, time_steps, sample_profile, df_household):
    # Deepcopy the network to ensure thread-safety
    net = net.deepcopy()

    # Select load buses
    load_bus_1 = net.load.index.intersection([0])
    load_buses_household = net.load.index.intersection([1, 2, 3, 4, 5])

    # Precompute household scaling factor
    household_profile = df_household['P_HOUSEHOLD'] / par.house_scaling

    # Collect results for this sample
    sample_results = {'loads': [], 'buses': [], 'lines': [], 'trafos': []}

    for t in time_steps:
        # Update load values directly
        net.load.loc[load_bus_1, 'p_mw'] = sample_profile.loc[t, 'meanP']
        net.load.loc[load_buses_household, 'p_mw'] = household_profile[t]

        # Run DC power flow with reduced logging
        pp.runpp(net, check_connectivity=False, verbose=False)

        # Collect results at each time step
        load_results = net.res_load[['p_mw']].copy()
        load_results['time_step'] = t
        
        bus_results = net.res_bus[['vm_pu', 'va_degree']].copy()
        bus_results['time_step'] = t
        
        line_results = net.res_line[['loading_percent', 'i_ka']].copy()
        line_results['time_step'] = t
        
        trafo_results = net.res_trafo[['loading_percent']].copy()
        trafo_results['time_step'] = t

        # Append time-step results to sample's results
        sample_results['loads'].append(load_results)
        sample_results['buses'].append(bus_results)
        sample_results['lines'].append(line_results)
        sample_results['trafos'].append(trafo_results)

    # Concatenate results from all time steps for this sample
    return (
        pd.concat(sample_results['loads'], ignore_index=True),
        pd.concat(sample_results['buses'], ignore_index=True),
        pd.concat(sample_results['lines'], ignore_index=True),
        pd.concat(sample_results['trafos'], ignore_index=True)
    )

def montecarlo_analysis_parallel(net, time_steps, df_season_heatpump_prognosis, df_household, n_jobs=4):
    mc_samples = generate_samples(df_season_heatpump_prognosis)  # Generate MC samples

    # Start timing
    start_time = time.time()

    # Use joblib with tqdm for parallel processing with progress reporting
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_sample)(net, time_steps, sample_profile, df_household) 
        for sample_profile in tqdm(mc_samples, desc="Processing samples")
    )

    # Calculate total elapsed time
    total_time = time.time() - start_time
    print(f"Monte Carlo analysis completed for {len(mc_samples)} samples in parallel.")
    print(f"Total time taken: {total_time:.2f} seconds.")
    
    return all_results

def run_single_sample_with_violation(
    net, time_steps, sample_profile, opf_results, const_load_household, heatpump_scaling_factors_df
):
    net = net.deepcopy()

    # Extract OPF results
    flexible_load_vars = opf_results['load']
    ts_in = opf_results['thermal_storage']['ts_in']
    ts_out = opf_results['thermal_storage']['ts_out']

    # Initialize violation counters
    line_violations = {}  # Store line violations: {line_idx: {timestep: count}}
    trafo_violations = {}  # Store transformer violations: {timestep: count}
    total_violations = 0

    # Results storage
    sample_results = {'loads': [], 'buses': [], 'lines': [], 'trafos': []}

    for t in time_steps:
        # Update fixed household loads using the ConstControl
        const_load_household.time_step(net, time=t)

        for load_index, scaling_data in heatpump_scaling_factors_df.iterrows():
            scaling_factor = scaling_data['p_mw']
            bus = scaling_data['bus']

            try:
                # Map Monte Carlo sample to heat pump load
                sampled_heat_demand = sample_profile['P_HEATPUMP_NORM'].loc[t] * scaling_factor

                # Compute adjusted load
                nominal_heat_demand = flexible_load_vars[t][bus]
                ts_out_value = ts_out[t][bus]
                ts_in_value = ts_in[t][bus]
                adjusted_load = nominal_heat_demand + (sampled_heat_demand - nominal_heat_demand)
                adjusted_load += (ts_out_value - ts_in_value) / par.COP

                # Update the load in the network
                net.load.at[load_index, 'p_mw'] = adjusted_load

            except Exception as e:
                print(f"Error updating load_index {load_index}, bus {bus} at time {t}: {e}")
                continue

        try:
            pp.rundcpp(net, check_connectivity=False, verbose=False)
        except pp.optimality.PandapowerRunError:
            total_violations += 1
            continue

        # Check for line violations
        for line_idx, loading in net.res_line['loading_percent'].items():
            if loading > 100:
                total_violations += 1
                if line_idx not in line_violations:
                    line_violations[line_idx] = {}
                line_violations[line_idx][t] = line_violations[line_idx].get(t, 0) + 1

        # Check for transformer violations
        for trafo_idx, loading in net.res_trafo['loading_percent'].items():
            if loading > 100:
                total_violations += 1
                trafo_violations[t] = trafo_violations.get(t, 0) + 1

        # Collect results
        load_results = net.res_load[['p_mw']].copy()
        load_results['time_step'] = t

        bus_results = net.res_bus[['vm_pu', 'va_degree']].copy()
        bus_results['time_step'] = t

        line_results = net.res_line[['loading_percent', 'i_ka']].copy()
        line_results['time_step'] = t

        trafo_results = net.res_trafo[['loading_percent']].copy()
        trafo_results['time_step'] = t

        sample_results['loads'].append(load_results)
        sample_results['buses'].append(bus_results)
        sample_results['lines'].append(line_results)
        sample_results['trafos'].append(trafo_results)

    return (
        pd.concat(sample_results['loads'], ignore_index=True),
        pd.concat(sample_results['buses'], ignore_index=True),
        pd.concat(sample_results['lines'], ignore_index=True),
        pd.concat(sample_results['trafos'], ignore_index=True),
        total_violations,
        line_violations,
        trafo_violations,
    )

def montecarlo_analysis_with_violations(
    net,
    time_steps,
    df_season_heatpump_prognosis,
    opf_results,
    const_load_household,
    heatpump_scaling_factors_df,
    n_jobs=-1,
    log_file="violation_log.txt",
):
    # Generate Monte Carlo samples for heat demand
    mc_samples = generate_samples(df_season_heatpump_prognosis)

    # Initialize log storage
    overall_line_violations = {}
    overall_trafo_violations = {}

    # Start timing
    start_time = time.time()

    # Use joblib with tqdm for parallel processing
    results_and_violations = Parallel(n_jobs=n_jobs)(
        delayed(run_single_sample_with_violation)(
            net,
            time_steps,
            sample_profile,
            opf_results,
            const_load_household,
            heatpump_scaling_factors_df,
        )
        for sample_profile in tqdm(mc_samples, desc="Processing samples")
    )

    # Process results
    all_results = [res[:-3] for res in results_and_violations]
    violation_counts = [res[-3] for res in results_and_violations]
    line_violations_list = [res[-2] for res in results_and_violations]
    trafo_violations_list = [res[-1] for res in results_and_violations]

    # Aggregate line and transformer violations
    for line_violations in line_violations_list:
        for line_idx, times in line_violations.items():
            if line_idx not in overall_line_violations:
                overall_line_violations[line_idx] = {}
            for t, count in times.items():
                overall_line_violations[line_idx][t] = overall_line_violations[line_idx].get(t, 0) + count

    for trafo_violations in trafo_violations_list:
        for t, count in trafo_violations.items():
            overall_trafo_violations[t] = overall_trafo_violations.get(t, 0) + count

    # Find maximum violations
    max_violations_line = max(
        ((line, t, count) for line, times in overall_line_violations.items() for t, count in times.items()),
        key=lambda x: x[2],
        default=(None, None, 0),
    )

    # Log violations to a file
    with open(log_file, "w") as f:
        f.write("Line Constraint Violations:\n")
        for line_idx, times in overall_line_violations.items():
            for t, count in times.items():
                f.write(f"Line {line_idx}: Time Step {t}, {count} violations\n")

        f.write("\nTransformer Constraint Violations:\n")
        for t, count in overall_trafo_violations.items():
            f.write(f"Time Step {t}, {count} violations\n")

        f.write("\nMaximum Violations:\n")
        f.write(f"Line {max_violations_line[0]}, Time Step {max_violations_line[1]}: {max_violations_line[2]} violations\n")

    # Calculate probability of constraint violation
    total_violations = sum(violation_counts)
    total_simulations = len(mc_samples) * len(time_steps)
    violation_probability = total_violations / total_simulations
    total_time = time.time() - start_time

    print(f"Monte Carlo analysis completed for {len(mc_samples)} samples in parallel.")
    print(f"Total time taken: {total_time:.2f} seconds.")
    print(f"Probability of constraint violation: {violation_probability:.4f}")
    print(f"Violation log saved to {log_file}")

    return all_results, violation_probability
