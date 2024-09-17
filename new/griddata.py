"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

Grid Data File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import pandapower.control as control
from pandapower.control import ConstControl
from pandapower.timeseries import OutputWriter, DFData

def setup_grid():
    net = pn.create_cigre_network_lv()

    # Switch off industrial and commercial loads
    net.switch.loc[1, "closed"] = False
    net.switch.loc[2, "closed"] = False

    # Iterate over all loads in the network and set controllable to False (i.e. not flexible)
    for load_idx in net.load.index:
        net.load.at[load_idx, 'controllable'] = False

    # Remove the switch between bus 0 and bus 1
    switch_to_remove = net.switch[(net.switch.bus == 0) & (net.switch.element == 1)].index
    net.switch.drop(switch_to_remove, inplace=True)
    
    # Change the transformer HV bus from bus 1 to bus 0
    net.trafo.at[0, 'hv_bus'] = 0

    # Remove bus 1 and any associated elements
    bus_to_remove = 1
    net.bus.drop(bus_to_remove, inplace=True)
    net.load = net.load[~net.load.bus.isin([bus_to_remove])]
    net.sgen = net.sgen[~net.sgen.bus.isin([bus_to_remove])]
    net.line = net.line[~net.line.from_bus.isin([bus_to_remove]) & ~net.line.to_bus.isin([bus_to_remove])]
    net.trafo = net.trafo[~net.trafo.hv_bus.isin([bus_to_remove]) & ~net.trafo.lv_bus.isin([bus_to_remove])]
    net.switch = net.switch[~net.switch.bus.isin([bus_to_remove]) & ~net.switch.element.isin([bus_to_remove])]

    # Renumber the remaining buses to fill the gap
    old_to_new_bus_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(net.bus.index))}
    net.bus.rename(index=old_to_new_bus_map, inplace=True)

    # Update all elements that reference bus indices
    def update_bus_references(df, columns):
        for col in columns:
            df[col] = df[col].map(old_to_new_bus_map)

    update_bus_references(net.load, ['bus'])
    update_bus_references(net.sgen, ['bus'])
    update_bus_references(net.line, ['from_bus', 'to_bus'])
    update_bus_references(net.trafo, ['hv_bus', 'lv_bus'])
    update_bus_references(net.switch, ['bus', 'element'])

    # Add PV generators to the corresponding buses
    pv_buses = [12, 16, 17, 18, 19]
    pv_generators = []

    # Load and preprocess the load profile CSV file
    df = pd.read_csv("load_profile_1111.csv")
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    df['time_step'] = range(len(df))  # Create a numerical index
    df.set_index('time_step', inplace=True)
    df['mult'] = df['mult'] * 15 / 1000
    ds = DFData(df)

    # Load and preprocess the PV generation profile CSV file
    df_pv = pd.read_csv("pv_generation_profile.csv")
    df_pv['time'] = pd.to_datetime(df_pv['time'], format='%H:%M:%S').dt.time
    df_pv['time_step'] = range(len(df_pv))  # Create a numerical index
    df_pv.set_index('time_step', inplace=True)
    df_pv['pvgen'] = df_pv['pvgen'] * 300 / 1000
    ds_pv = DFData(df_pv)

    for bus in pv_buses:
        pv_gen = pp.create_sgen(net, old_to_new_bus_map[bus], p_mw=0, q_mvar=0, type='pv', controllable=True)
        # Set initial limits
        net.sgen.at[pv_gen, 'min_p_mw'] = 0
        net.sgen.at[pv_gen, 'max_p_mw'] = df_pv['pvgen'].max()  # Set this to the maximum possible value in the profile
        net.sgen.at[pv_gen, 'min_q_mvar'] = -0.5 / 1000  # Example value, adjust as needed
        net.sgen.at[pv_gen, 'max_q_mvar'] = 0.5 / 1000  # Example value, adjust as needed
        pv_generators.append(pv_gen)

    # Add the Load profile to the network
    profile_loads = net.load.index.intersection([0, 1, 2, 3, 4, 5])
    const_load = ConstControl(net, element='load', element_index=profile_loads,
                              variable='p_mw', data_source=ds, profile_name=["mult"] * len(profile_loads))

    # Initialize ConstControl for PV and load profiles with correct time steps
    const_pv = ConstControl(net, element='sgen', element_index=pv_generators,
                            variable='p_mw', data_source=ds_pv, profile_name=["pvgen"] * len(pv_generators))

    # Remove buses with prefixes "I" or "C" along with associated elements
    buses_to_remove = net.bus[net.bus['name'].str.startswith(('Bus I', 'Bus C'))].index

    if not buses_to_remove.empty:
        # Drop the buses
        net.bus.drop(buses_to_remove, inplace=True)
        
        # Remove loads, sgens, lines, transformers connected to those buses
        net.load = net.load[~net.load.bus.isin(buses_to_remove)]
        net.sgen = net.sgen[~net.sgen.bus.isin(buses_to_remove)]
        net.line = net.line[~net.line.from_bus.isin(buses_to_remove) & ~net.line.to_bus.isin(buses_to_remove)]
        net.trafo = net.trafo[~net.trafo.hv_bus.isin(buses_to_remove) & ~net.trafo.lv_bus.isin(buses_to_remove)]
    
    # Remove switches associated with the deleted buses
    switches_to_remove = net.switch[(net.switch.bus.isin(buses_to_remove)) | (net.switch.element.isin(buses_to_remove))].index
    net.switch.drop(switches_to_remove, inplace=True)
    
    return net, df_pv, df, pv_generators, const_load, const_pv


def setup_grid2():
    
    net = pn.panda_four_load_branch()

    # Add PV generators to the corresponding buses
    pv_buses = [3]
    pv_generators = []

    # Load and preprocess the load profile CSV file
    df = pd.read_csv("load_profile_1111.csv")
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    df['time_step'] = range(len(df))  # Create a numerical index
    df.set_index('time_step', inplace=True)
    df['mult'] = df['mult'] * 15 / 1000
    ds = DFData(df)

    # Load and preprocess the PV generation profile CSV file
    df_pv = pd.read_csv("pv_generation_profile.csv")
    df_pv['time'] = pd.to_datetime(df_pv['time'], format='%H:%M:%S').dt.time
    df_pv['time_step'] = range(len(df_pv))  # Create a numerical index
    df_pv.set_index('time_step', inplace=True)
    df_pv['pvgen'] = df_pv['pvgen'] * 300 / 1000
    ds_pv = DFData(df_pv)

    for bus in pv_buses:
        pv_gen = pp.create_sgen(net, bus, p_mw=0, q_mvar=0, type='pv', controllable=True)
        # Set initial limits
        net.sgen.at[pv_gen, 'min_p_mw'] = 0
        net.sgen.at[pv_gen, 'max_p_mw'] = df_pv['pvgen'].max()  # Set this to the maximum possible value in the profile
        net.sgen.at[pv_gen, 'min_q_mvar'] = -0.5 / 1000  # Example value, adjust as needed
        net.sgen.at[pv_gen, 'max_q_mvar'] = 0.5 / 1000  # Example value, adjust as needed
        pv_generators.append(pv_gen)

    # Add the Load profile to the network
    profile_loads = net.load.index.intersection([0, 1, 2, 3, 4, 5])
    const_load = ConstControl(net, element='load', element_index=profile_loads,
                              variable='p_mw', data_source=ds, profile_name=["mult"] * len(profile_loads))

    # Initialize ConstControl for PV and load profiles with correct time steps
    const_pv = ConstControl(net, element='sgen', element_index=pv_generators,
                            variable='p_mw', data_source=ds_pv, profile_name=["pvgen"] * len(pv_generators))
    
    return net, df_pv, df, pv_generators, const_load, const_pv