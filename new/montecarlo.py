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
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter
from joblib import Parallel, delayed
from tqdm import tqdm

#### SCRIPTS ####
import griddata as gd
import opf as opf
import results as rs
import plot as pl
import drcc
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
        df_sample['meanP'] = df_sample['meanP'] / par.hp_scaling
        
        # Append to the list
        mc_samples.append(df_sample)
    
    # Print the header of the first sample for inspection
    print("Header of first sample:\n", mc_samples[0].head())
    
    return mc_samples

###############################################################################
## PANDAPOWER FUNCTIONS ##
###############################################################################
def create_output_writer(net, time_steps, output_dir, sample_index):
    # Create a unique output directory or file for each Monte Carlo sample
    sample_output_dir = f"{output_dir}/sample_{sample_index}"
    ow = OutputWriter(net, time_steps, output_path=sample_output_dir, output_file_type=".xlsx", log_variables=list())
    
    # Log variables for loads, buses, lines, and transformers
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    ow.log_variable('res_trafo', 'loading_percent')  # Log transformer loading
    
    return ow

###############################################################################
## MONTECARLO ANALYSIS ##
###############################################################################


def montecarlo_analysis(net, time_steps, df_season_heatpump_prognosis, df_household):
    mc_samples = generate_samples(df_season_heatpump_prognosis)  # Generate MC samples
    all_results = []  # List to collect results from all samples

    # Iterate over each sample profile
    for idx, sample_profile in enumerate(mc_samples):
        print(f"Processing sample {idx}")

        # Drop any controllers (no longer needed)
        net.controller.drop(net.controller.index, inplace=True)

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

            # Run DC power flow
            pp.rundcpp(net, check_connectivity=False, verbose=False)

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

        # Concatenate results from all time steps for this sample and store in `all_results`
        all_results.append((
            pd.concat(sample_results['loads'], ignore_index=True),
            pd.concat(sample_results['buses'], ignore_index=True),
            pd.concat(sample_results['lines'], ignore_index=True),
            pd.concat(sample_results['trafos'], ignore_index=True)
        ))
    
    print(f"Monte Carlo analysis completed for {len(mc_samples)} samples.")
    
    return all_results

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