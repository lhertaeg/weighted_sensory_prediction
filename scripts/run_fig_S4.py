#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:50:33 2023

@author: loreen.hertaeg
"""

# %% Import

import pickle
import numpy as np

from src.functions_simulate import simulate_example_pe_circuit_cntns
from src.plot_data import plot_example_stimuli_smoothed, plot_dev_cntns, plot_example_mean, plot_example_variance

dtype = np.float32


# %% Run network for filtered (smoothed) continuous input

run_cell = False
plot_only = True

if run_cell:
    
    ### filename for data
    file_for_example = '../results/data/moments/data_net_example_cntns_input_one_column.pickle'
    file_for_data = '../results/data/moments/data_cntns_input_one_column.pickle'
    
    if not plot_only:
    
        ### global parameters
        trial_duration = np.int32(200000)
        num_values_per_trial = np.int32(400)
        n_ss = 3 * trial_duration//4
        
        ### smoothing parameters to test
        hann_windows = np.int32(np.linspace(25,500,6))
        
        ### initialise
        dev_mean = np.zeros_like(hann_windows, dtype=dtype)
        dev_variance = np.zeros_like(hann_windows, dtype=dtype)
        
        ### run experiments
        for i, hann_window in enumerate(hann_windows):
        
            [_, _, trial_duration, _, stimuli, 
             m_neuron, v_neuron] = simulate_example_pe_circuit_cntns('10', 5, 2, file_for_data, trial_duration = trial_duration, 
                                                                     num_values_per_trial = num_values_per_trial, hann_window=hann_window)
                                                                     
            running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1, dtype=dtype)
            dev_mean[i] = (np.mean(running_average[n_ss:]) - np.mean(m_neuron[n_ss:])) / np.mean(running_average[n_ss:])
            
            mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1, dtype=dtype)
            momentary_variance = (stimuli - mean_running)**2
            running_variance = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1, dtype=dtype)
            dev_variance[i] = (np.mean(running_variance[n_ss:]) - np.mean(v_neuron[n_ss:])) / np.mean(running_variance[n_ss:])
        
        ### save data
        with open(file_for_example,'wb') as f:
            pickle.dump([stimuli, trial_duration, m_neuron, v_neuron], f)
            
        with open(file_for_data,'wb') as f:
            pickle.dump([hann_windows, dev_mean, dev_variance], f)
            
    else:
        
        ### load data
        with open(file_for_example,'rb') as f:
            [stimuli, trial_duration, m_neuron, v_neuron] = pickle.load(f)
            
        with open(file_for_data,'rb') as f:
            [hann_windows, dev_mean, dev_variance] = pickle.load(f)
    
    ### plot data
    plot_example_stimuli_smoothed([hann_windows[0], hann_windows[-1]])

    plot_dev_cntns(hann_windows, dev_mean, dev_variance)

    plot_example_mean(stimuli, trial_duration, m_neuron, mse_flg=False)
    plot_example_variance(stimuli, trial_duration, v_neuron, mse_flg=False)