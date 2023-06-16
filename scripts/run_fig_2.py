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
from src.plot_data import plot_example_mean, plot_example_variance, plot_mse_heatmap, plot_interneuron_activity_heatmap
from src.plot_data import plot_interneuron_activity

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Example: estimating mean and variance through PE neuron activity

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
    file_for_data = '../results/data/moments/data_example_mfn_' + mfn_flag + '.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        [_, _, trial_duration, _, stimuli, m_neuron, v_neuron] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_for_data)
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [_, _, trial_duration, _, stimuli, m_neuron, v_neuron] = pickle.load(f)
            
    ### plot single panels
    plot_example_mean(stimuli, trial_duration, m_neuron, figsize=(5,3), fs=7, lw=1.2)
    plot_example_variance(stimuli, trial_duration, v_neuron, figsize=(5,3), fs=7, lw=1.2)


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
        means_tested = np.linspace(3,6,7, dtype=dtype)
        variances_tested = np.linspace(2,9,8, dtype=dtype)
        
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
                      title='M neuron encodes mean for a wide range of parameters', 
                      figsize=(4,3), fs=7, x_example=5, y_example=2**2) # vmax=0.3
    plot_mse_heatmap(end_of_initial_phase, means_tested, variances_tested, mse_variance, 
                      title='V neuron encodes variance for a wide range of parameters', show_mean=False, 
                      figsize=(4,3), fs=7, x_example=5, y_example=2**2, flg_var=True, digits_round=1)#, vmax=10)


# %% How do the interneurons behave with different input statistics? (later you could combine it with the one above!)

run_cell = True
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_interneuron_activity_heatmap_mfn_' + mfn_flag + '.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
    
        ## define parameters ranges tested:
        means_tested = np.linspace(3,6,7, dtype=dtype)
        variances_tested = np.linspace(2,9,8, dtype=dtype)
        
        ## run simulations
        [trial_duration, num_values_per_trial, means_tested, variances_tested, 
         mse_mean, mse_variance, activity_interneurons] = simulate_pe_uniform_para_sweep(mfn_flag, means_tested, variances_tested, 
                                                                                         file_for_data, record_interneuron_activity = True)
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [trial_duration, num_values_per_trial, means_tested, 
             variances_tested, mse_mean, mse_variance, activity_interneurons] = pickle.load(f)
        
    ### plot 
    end_of_initial_phase = np.int32(trial_duration * 0.5)
    #plot_interneuron_activity_heatmap(end_of_initial_phase, means_tested, variances_tested, activity_interneurons)
    plot_interneuron_activity(end_of_initial_phase, means_tested, variances_tested, activity_interneurons) # id_fixed=0 # play with id_fixed if wanted

