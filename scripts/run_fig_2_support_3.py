#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:50:33 2023

@author: loreen.hertaeg
"""

# %% Import

import pickle
import numpy as np

from src.functions_simulate import simulate_spatial_example
from src.plot_data import plot_examples_spatial_M, plot_examples_spatial_V, plot_deviation_spatial

dtype = np.float32

# %% Test several input statistcis

run_cell = True
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_spatial_mfn_diff_input_statistics.pickle'
    file_temporary = '../results/data/moments/data_spatial_mfn_temporary.pickle'
    
    ### input statistic parameterisation
    means_tested = np.linspace(3,6,7)
    stds_tested = np.sqrt(np.linspace(3,6,7, dtype=dtype))
    
    ### initialise
    deviation_mean = np.zeros((len(means_tested), len(stds_tested)))
    deviation_std = np.zeros((len(means_tested), len(stds_tested)))
    
    ### get data
    if not plot_only: # simulate respective network
        
        for i, mean in enumerate(means_tested):
            for j, std in enumerate(stds_tested):
        
                print('Mean no.: ' , (i+1) , '/' , len(means_tested), ', Std no.: ', (j+1), '/', len(stds_tested))
                [mean_stimulus, spatial_std, spatial_noise, 
                 num_time_steps, m_neuron, v_neuron, rates_final] = simulate_spatial_example(mfn_flag, dtype(mean), dtype(std), file_temporary)
                                                                                             
                deviation_mean[i,j] = (m_neuron[-1] - mean) / mean
                deviation_std[i,j] = (v_neuron[-1] - std**2) / std**2
                
        ### save data
        with open(file_for_data,'wb') as f:
            pickle.dump([means_tested, stds_tested, deviation_mean, deviation_std],f)
                                                                                        
    else: # load results

        with open(file_for_data,'rb') as f:
            [means_tested, stds_tested, deviation_mean, deviation_std] = pickle.load(f)

    ### plot data
    x_examples = [4, 4, 6]
    y_examples = [4, 6, 4]
    markers_examples = ['o', '^', '^']
    plot_deviation_spatial(deviation_mean, means_tested, stds_tested, vmin=0.35, vmax=1.2, fs=6, ax=None,
                           x_examples = x_examples, y_examples = y_examples, markers_examples = markers_examples)
    plot_deviation_spatial(deviation_std, means_tested, stds_tested, vmin=4.2, vmax=4.4, fs=6, show_mean=False, ax=None,
                           x_examples = x_examples, y_examples = y_examples, markers_examples = markers_examples)


# %% Transition to new mean

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_example_spatial_mfn_' + mfn_flag + '_diff_means.pickle'
    
    ### load data before statistic change
    file_before = '../results/data/moments/data_example_spatial_mfn_' + mfn_flag + '.pickle'
    with open(file_before,'rb') as f:
        [mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron_before, v_neuron_before, rates_final_before] = pickle.load(f)
    
    ### get data
    if not plot_only: # simulate respective network
        
        [mean_stimulus, spatial_std, spatial_noise, 
         num_time_steps, m_neuron, v_neuron, rates_final] = simulate_spatial_example(mfn_flag, dtype(6), dtype(2), file_for_data, 
                                                                        M_init = m_neuron_before[-1], V_init = v_neuron_before[-1], 
                                                                        rates_init = rates_final_before)
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron, v_neuron, rates_final] = pickle.load(f)


    plot_examples_spatial_M(num_time_steps, m_neuron_before, m_neuron, 4, 6, labels=['Before','After'])
    plot_examples_spatial_V(num_time_steps, v_neuron_before, v_neuron, 4, 4, labels=['Before','After'])


# %% Transition to new noise level

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_example_spatial_mfn_' + mfn_flag + '_diff_noise_levels.pickle'
    
    ### load data before statistic change
    file_before = '../results/data/moments/data_example_spatial_mfn_' + mfn_flag + '.pickle'
    with open(file_before,'rb') as f:
        [mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron_before, v_neuron_before, rates_final_before] = pickle.load(f)
    
    ### get data
    if not plot_only: # simulate respective network
        
        [mean_stimulus, spatial_std, spatial_noise, 
         num_time_steps, m_neuron, v_neuron, rates_final] = simulate_spatial_example(mfn_flag, dtype(4), dtype(np.sqrt(6)), file_for_data, 
                                                                        M_init = m_neuron_before[-1], V_init = v_neuron_before[-1], 
                                                                        rates_init = rates_final_before)
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron, v_neuron, rates_final] = pickle.load(f)


    plot_examples_spatial_M(num_time_steps, m_neuron_before, m_neuron, 4, 4, labels=['Before','After'])
    plot_examples_spatial_V(num_time_steps, v_neuron_before, v_neuron, 4, 6, labels=['Before','After'])


# %% Spatial instead of temporal noise - example

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_example_spatial_mfn_' + mfn_flag + '.pickle'
    
    ### get data
    if not plot_only: # simulate respective network
        
        [mean_stimulus, spatial_std, spatial_noise, 
         num_time_steps, m_neuron, v_neuron, rates_final] = simulate_spatial_example(mfn_flag, dtype(4), dtype(2), file_for_data)
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron, v_neuron, rates_final] = pickle.load(f)

