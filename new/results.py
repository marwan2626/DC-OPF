"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

Results File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####
import ast
import pandas as pd 

# Check and convert string representations of lists back to actual lists if necessary
def convert_to_list_if_needed(cell):
    if isinstance(cell, str):
        return eval(cell)
    return cell
import pandas as pd


def load_results(results_df):
    theta_degrees = results_df['theta_degrees']
    loading_percent = results_df['line_loading_percent']
    load_p_mw = results_df['load_p_mw']
    sgen_p_mw = results_df['sgen_p_mw']
    line_pl_mw = results_df['line_pl_mw']
    i_ka = results_df['line_current_mag']

    return theta_degrees, loading_percent, load_p_mw, sgen_p_mw, line_pl_mw, i_ka

def load_results_opf(results_opf_df):
    loading_percent = results_opf_df['line_loading_percent']
    ext_grid_p_mw = results_opf_df['ext_grid_p_mw']
    sgen_p_mw = results_opf_df['sgen_p_mw']
    curtailment_pv_mw = results_opf_df['curtailment_pv_mw']
    load_p_mw = results_opf_df['load_p_mw']

    return  loading_percent, ext_grid_p_mw, sgen_p_mw, curtailment_pv_mw, load_p_mw