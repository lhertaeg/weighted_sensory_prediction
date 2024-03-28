#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:20:45 2024

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal

from src.functions_simulate import simulate_example_pe_circuit, simulate_pe_uniform_para_sweep, simulate_effect_baseline
from src.functions_simulate import simulate_example_pe_circuit_cntns
from src.plot_data import plot_example_mean, plot_example_variance, plot_mse_heatmap, plot_neuron_activity

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% run example to show how nPE and pPE establishes M robustly (later combine with first cell in run_fig_2 MAYBE)

run_cell = False
plot_only = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_PE_establish_M_' + mfn_flag + '.pickle'
    
    ### flag
    save_PE = True
    
    ### get data
    if not plot_only: # simulate respective network
        
        if save_PE:
            [_, _, trial_duration, _, stimuli, m_neuron, v_neuron, PE] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_for_data, save_PE=save_PE,
                                                                                                     trial_duration = np.int32(200000), 
                                                                                                     num_values_per_trial = np.int32(400))
        else:
            [_, _, trial_duration, _, stimuli, m_neuron, v_neuron] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_for_data, save_PE=save_PE)
    
    else: # load results from previous simulation
        
        if save_PE:
            with open(file_for_data,'rb') as f:
                [_, _, trial_duration, _, stimuli, m_neuron, v_neuron, PE] = pickle.load(f)
        else:
            with open(file_for_data,'rb') as f:
                [_, _, trial_duration, _, stimuli, m_neuron, v_neuron] = pickle.load(f)
            
    ### plot single panels
    plot_example_mean(stimuli, trial_duration, m_neuron, figsize=(5,3), fs=7, lw=1.2)
    # plot_example_variance(stimuli, trial_duration, v_neuron, figsize=(5,3), fs=7, lw=1.2)
    
    plt.figure()
    plt.plot(PE[:,0])
    plt.plot(PE[:,1])
    
    
# %% Investigate/Illustrate effect of BL for one column

run_cell = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_temp.pickle'
    
    ### define the steady state
    trial_duration = np.int32(130000)
    n_ss = 3 * trial_duration//4
    
    ### define inputs and initialise
    add_ins = np.linspace(0,0.5,5)
    M_nPE, M_pPE, M_both = np.zeros_like(add_ins), np.zeros_like(add_ins), np.zeros_like(add_ins)
    V_nPE, V_pPE, V_both = np.zeros_like(add_ins), np.zeros_like(add_ins), np.zeros_like(add_ins)
    BL_nPE, BL_pPE, BL_both = np.zeros_like(add_ins), np.zeros_like(add_ins), np.zeros_like(add_ins)
    
    ### run network with BL increased in nPE neurons
    for i, add_in in enumerate(add_ins):
        
        add_input = np.zeros(8)
        add_input[0] = add_in
        
        ### run network
        [_, _, _, _, _, m_neuron, v_neuron, PE] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_for_data, add_input = add_input,
                                                                              trial_duration=trial_duration, save_PE=True)
        ### extract final M estimate
        M_nPE[i] = np.mean(m_neuron[n_ss:])
        V_nPE[i] = np.mean(v_neuron[n_ss:])
        BL_nPE[i] = np.mean(PE[np.where(abs(np.diff(PE[:,0]))<1e-6)[0],0])
        
    
    ### run network with BL increased in pPE neurons
    for i, add_in in enumerate(add_ins):
        
        add_input = np.zeros(8)
        add_input[1] = add_in
        
        ### run network
        [_, _, _, _, _, m_neuron, v_neuron, PE] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_for_data, add_input = add_input,
                                                                              trial_duration=trial_duration, save_PE=True)
        ### extract final M estimate
        M_pPE[i] = np.mean(m_neuron[n_ss:])
        V_pPE[i] = np.mean(v_neuron[n_ss:])
        BL_pPE[i] = np.mean(PE[np.where(abs(np.diff(PE[:,1]))<1e-6)[0],1])
        

    ### run network with BL increased in both PE neurons
    for i, add_in in enumerate(add_ins):
        
        add_input = np.zeros(8)
        add_input[:2] = add_in
        
        ### run network
        [_, _, _, _, _, m_neuron, v_neuron, PE] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_for_data, add_input = add_input,
                                                                              trial_duration=trial_duration, save_PE=True)
        ### extract final M estimate
        M_both[i] = np.mean(m_neuron[n_ss:])
        V_both[i] = np.mean(v_neuron[n_ss:])
        BL_both[i] = np.mean(PE[np.where(abs(np.diff(PE[:,1]))<1e-6)[0],1])


    file_for_data = '../results/data/moments/data_influence_of_baseline.pickle'
    with open(file_for_data,'wb') as f:
        pickle.dump([M_nPE, M_pPE, M_both, V_nPE, V_pPE, V_both, BL_nPE, BL_pPE, BL_both], f)
        
        
    plt.figure()
    plt.plot(BL_nPE, M_nPE)
    plt.plot(BL_pPE, M_pPE)
    plt.plot(BL_both, M_both)
    
    plt.figure()
    plt.plot(BL_nPE, V_nPE)
    plt.plot(BL_pPE, V_pPE)
    plt.plot(BL_both, V_both)
        
    
    
# %% Investigate/Illustrate effect of BL ffor full network (lower/higher subnetwork)

run_cell = False

if run_cell:

    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11

    ### define target area and stimulus statistics
    column = 0      # 0: both columns, 1: lower level PE circuit, 2: higher level PE circuit
    std_mean = 1    # [0,1]  
    n_std = 0       # [0,1]
 
    ### get data
    if not plot_only: # simulate respective network

        ### define the baselines to be tested
        baselines = np.linspace(0,0.5,5)
        
        ### run for nPE neuron
        cell_id = 0
        sensory_weights_nPE = simulate_effect_baseline(mfn_flag, std_mean, n_std, column, baselines, cell_id)
        
        ### run for pPE neuron
        cell_id = 1
        sensory_weights_pPE = simulate_effect_baseline(mfn_flag, std_mean, n_std, column, baselines, cell_id)
        
        ### run for both PE neurons
        cell_id = np.array([0,1])
        sensory_weights_both = simulate_effect_baseline(mfn_flag, std_mean, n_std, column, baselines, cell_id)

        ### save data
        file_for_data = '../results/data/moments/data_influence_of_baseline_full_' + str(std_mean) + '_' + str(n_std) + '.pickle'
        with open(file_for_data,'wb') as f:
            pickle.dump([baselines, sensory_weights_nPE, sensory_weights_pPE, sensory_weights_both], f)

    else:

        # with open(file_for_data,'rb') as f:
        #     [pert_strength, m_act_lower, v_act_lower, v_act_higher] = pickle.load(f)
        print('tbc')

    ### plot data  
    plt.figure()
    plt.plot(sensory_weights_nPE)
    plt.plot(sensory_weights_pPE)
    plt.plot(sensory_weights_both)
    
    
# %% Run network for filtered (smoothed) continuous input

run_cell = False
plot_only = False

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
    plot_example_mean(stimuli, trial_duration, m_neuron, figsize=(5,3), fs=7, lw=1.2)
    plot_example_variance(stimuli, trial_duration, v_neuron, figsize=(5,3), fs=7, lw=1.2)
    
    plt.figure()
    plt.plot(hann_windows, dev_mean)
    
    plt.figure()
    plt.plot(hann_windows, dev_variance)
    
    
# %% filtering

# x = np.arange(300)
# y = np.zeros_like(x)
# y[100:200] = 1

# #z = np.zeros_like(x[:len(x)//2])
# mu = 150
# sig2 = 25
# z = np.exp(-0.5*(x - mu)**2/sig2) / np.sqrt(2* sig2 * np.pi)

# filtered = signal.convolve(y, z, mode='same')

# win = signal.windows.hann(25)
# #win = signal.windows.gaussian(150,30)
# filtered1 = signal.convolve(y, win, mode='same') / sum(win)

# # plt.figure()
# # plt.plot(win)
# # plt.plot(win1)

# plt.figure()
# plt.plot(x,y)
# plt.plot(x, filtered)
# plt.plot(x, filtered1)

# def random_uniform_from_moments(mean, sd, num):
    
#     b = np.sqrt(12) * sd / 2 + mean
#     a = 2 * mean - b
#     rnd = dtype(np.random.uniform(a, b, size = num))
        
#     return rnd


# trial_duration = np.int32(100000)
# mean_stimuli = 5
# std_stimuli = 2

# num_values_per_trial = np.int32(200)
# repeats_per_value = trial_duration//num_values_per_trial
# stimuli = random_uniform_from_moments(mean_stimuli, std_stimuli, num_values_per_trial)
# stimuli = np.repeat(stimuli, repeats_per_value)

# win = signal.windows.hann(800) # 25, 50, 100, 200, 400, 800
# stimuli_smoothed = signal.convolve(stimuli, win, mode='same') / sum(win)

# plt.figure()
# plt.plot(stimuli, color='k')
# plt.plot(stimuli_smoothed, color='r')


