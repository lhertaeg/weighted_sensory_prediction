#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_example_pe_circuit, simulate_pe_uniform_para_sweep
from src.plot_data import plot_example_mean, plot_example_variance, plot_mse_heatmap

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Example: estimating mean and variance through PE neuron activity

# Todo:
    # choose parameters consistent with the subsequent plots
    # beautify plotting functions
    # rerun (you have to because something in the functions changed!)

run_cell = False
plot_only = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_example_mfn_' + mfn_flag + '.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        [_, _, trial_duration, _, stimuli, m_neuron, v_neuron] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_for_data)
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [_, _, trial_duration, _, stimuli, m_neuron, v_neuron] = pickle.load(f)
            
    ### plot single panels
    plot_example_mean(stimuli, trial_duration, m_neuron)
    plot_example_variance(stimuli, trial_duration, v_neuron)


# %% Systematically run network for different parameterisations of uniform distribution
# record MSE between running estimate and m- or v-neuron output

# Todo:
    # choose parameters consistent with the subsequent plots
    # beautify plotting functions
    # rerun (you have to because something in the functions changed!)

run_cell = False
plot_only = True

if run_cell: 
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_heatmap_mfn_' + mfn_flag + '.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
    
        ## define parameters ranges tested:
        means_tested = np.linspace(3,6,7)
        variances_tested = np.linspace(2,9,8)
        
        ## run simulations
        [trial_duration, num_values_per_trial, means_tested, variances_tested, mse_mean, 
          mse_variance] = simulate_pe_uniform_para_sweep(mfn_flag, means_tested, variances_tested, file_for_data)
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [trial_duration, num_values_per_trial, means_tested, 
              variances_tested, mse_mean, mse_variance] = pickle.load(f)
            
    ### plot heatmaps
    end_of_initial_phase = np.int32(trial_duration * 0.5)
    plot_mse_heatmap(end_of_initial_phase, means_tested, variances_tested, mse_mean, 
                      title='Estimating the mean', vmax=0.3)
    plot_mse_heatmap(end_of_initial_phase, means_tested, variances_tested, mse_variance, 
                      title='Estimating the variance', show_mean=False)#, vmax=10)
    
    