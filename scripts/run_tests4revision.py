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
from src.functions_simulate import simulate_example_pe_circuit_cntns, random_uniform_from_moments, stimuli_moments_from_uniform
from src.plot_data import plot_example_mean, plot_example_variance, plot_mse_heatmap, plot_neuron_activity

from src.default_parameters import default_para_mfn
from src.functions_networks import run_mfn_circuit, run_mfn_circuit_coupled

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% run example to show how nPE and pPE establishes M robustly (later combine with first cell in run_fig_2 MAYBE)

run_cell = False
plot_only = True

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
plot_only = True

if run_cell:
    
    file_for_data = '../results/data/moments/data_influence_of_baseline.pickle'
    
    if not plot_only:
        ### choose mean-field network to simulate
        mfn_flag = '10' # valid options are '10', '01', '11
        
        ### filename for data
        file_temp = '../results/data/moments/data_temp.pickle'
        
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
            [_, _, _, _, _, m_neuron, v_neuron, PE] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_temp, add_input = add_input,
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
            [_, _, _, _, _, m_neuron, v_neuron, PE] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_temp, add_input = add_input,
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
            [_, _, _, _, _, m_neuron, v_neuron, PE] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_temp, add_input = add_input,
                                                                                  trial_duration=trial_duration, save_PE=True)
            ### extract final M estimate
            M_both[i] = np.mean(m_neuron[n_ss:])
            V_both[i] = np.mean(v_neuron[n_ss:])
            BL_both[i] = np.mean(PE[np.where(abs(np.diff(PE[:,1]))<1e-6)[0],1])
    
    
        
        with open(file_for_data,'wb') as f:
            pickle.dump([M_nPE, M_pPE, M_both, V_nPE, V_pPE, V_both, BL_nPE, BL_pPE, BL_both], f)
            
    else:
        
        with open(file_for_data,'rb') as f:
            [M_nPE, M_pPE, M_both, V_nPE, V_pPE, V_both, BL_nPE, BL_pPE, BL_both] = pickle.load(f)
        
    plt.figure()
    plt.plot(BL_nPE, M_nPE)
    plt.plot(BL_pPE, M_pPE)
    plt.plot(BL_both, M_both)
    
    plt.figure()
    plt.plot(BL_nPE, V_nPE)
    plt.plot(BL_pPE, V_pPE)
    plt.plot(BL_both, V_both)
        
    
# %% Investigate/Illustrate effect of BL for full network (lower/higher subnetwork)

run_cell = False
plot_only = True

if run_cell:

    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11

    ### define target area and stimulus statistics
    column = 0      # 0: both columns, 1: lower level PE circuit, 2: higher level PE circuit
    std_mean = 1    # [0,1]  
    n_std = 0       # [0,1]
    
    file_for_data = '../results/data/moments/data_influence_of_baseline_full_' + str(std_mean) + '_' + str(n_std) + '.pickle'
 
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
        with open(file_for_data,'wb') as f:
            pickle.dump([baselines, sensory_weights_nPE, sensory_weights_pPE, sensory_weights_both], f)

    else:

        with open(file_for_data,'rb') as f:
            [baselines, sensory_weights_nPE, sensory_weights_pPE, sensory_weights_both] = pickle.load(f)

    ### plot data  
    plt.figure()
    plt.plot(sensory_weights_nPE)
    plt.plot(sensory_weights_pPE)
    plt.plot(sensory_weights_both)
    
    
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
    plot_example_mean(stimuli, trial_duration, m_neuron, figsize=(5,3), fs=7, lw=1.2)
    plot_example_variance(stimuli, trial_duration, v_neuron, figsize=(5,3), fs=7, lw=1.2)
    
    plt.figure()
    plt.plot(hann_windows, dev_mean)
    
    plt.figure()
    plt.plot(hann_windows, dev_variance)
    
   
# %% Robustness: tau_V

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_change_tauV.pickle'
    
    ### get data
    if not plot_only: # simulate respective networks
    
        ## parameters tested
        paras = [1000, 9000]
    
        ## load default parameters
        VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
         v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
        
        ### one column
        
        ## generate stimuli
        np.random.seed(186)
        trial_duration_1 = np.int32(100000)
        num_values_per_trial_1 = np.int32(200)
        
        repeats_per_value = trial_duration_1//num_values_per_trial_1
        
        stimuli_1 = random_uniform_from_moments(5, 2, num_values_per_trial_1)
        stimuli_1 = np.repeat(stimuli_1, repeats_per_value)
        
        ## run network - parameter smaller 
        tc_var_per_stim = dtype(paras[0])
        m_neuron_s, v_neuron_s, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                        fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
        
        ## run network - parameter larger
        tc_var_per_stim = dtype(paras[1])
        m_neuron_l, v_neuron_l, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                        fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
        
        ### two columns
        
        ## generate stimuli
        np.random.seed(186)
        n_trials = np.int32(100)
        trial_duration_2 = np.int32(5000)
        num_values_per_trial_2 = np.int32(10)
        
        n_repeats_per_stim = dtype(trial_duration_2/num_values_per_trial_2)

        stimuli_2 = stimuli_moments_from_uniform(n_trials, num_values_per_trial_2, dtype(5 - np.sqrt(3)*5), 
                                               dtype(5 + np.sqrt(3)*5), dtype(0), dtype(5))
        stimuli_2 = np.repeat(stimuli_2, n_repeats_per_stim)
        
        
        ## run network - parameter smaller 
        tc_var_per_stim = dtype(paras[0])
        tc_var_pred = dtype(paras[0])
        [_, _, _, _, alpha_s, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                              v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                              tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                              VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                              v_PE_to_V = v_PE_to_V) 
        
        ## run network - parameter larger 
        tc_var_per_stim = dtype(paras[1])
        tc_var_pred = dtype(paras[1])
        [_, _, _, _, alpha_l, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                              v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                              tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                              VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                              v_PE_to_V = v_PE_to_V) 
        
    
        ## save
        with open(file_for_data,'wb') as f:
            pickle.dump([paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
                         m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l],f)
    
    else: # load results from previous simulation
        
        with open(file_for_data,'rb') as f:
            [paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
             m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
            
            
    ### plot single panels
    plot_example_mean(stimuli_1, trial_duration_1, m_neuron_s, figsize=(5,3), fs=7, lw=1.2)
    plot_example_variance(stimuli_1, trial_duration_1, v_neuron_s, figsize=(5,3), fs=7, lw=1.2)
    
    plot_example_mean(stimuli_1, trial_duration_1, m_neuron_l, figsize=(5,3), fs=7, lw=1.2)
    plot_example_variance(stimuli_1, trial_duration_1, v_neuron_l, figsize=(5,3), fs=7, lw=1.2)
    
    plt.figure()
    plt.plot(alpha_s, 'b')
    plt.plot(alpha_l, 'r')

   
# %% Robustness: wPE2P

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_change_wPE2P.pickle'
    
    ### get data
    if not plot_only: # simulate respective networks
    
        ## parameters tested
        paras = [0.5, 1.5]
    
        ## load default parameters
        VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
         v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
        
        ### one column
        
        ## generate stimuli
        np.random.seed(186)
        trial_duration_1 = np.int32(100000)
        num_values_per_trial_1 = np.int32(200)
        
        repeats_per_value = trial_duration_1//num_values_per_trial_1
        
        stimuli_1 = random_uniform_from_moments(5, 2, num_values_per_trial_1)
        stimuli_1 = np.repeat(stimuli_1, repeats_per_value)
        
        ## run network - parameter smaller 
        w_PE_to_P_mod = paras[0] * w_PE_to_P
        m_neuron_s, v_neuron_s, _ = run_mfn_circuit(w_PE_to_P_mod, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                    fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
        
        ## run network - parameter larger
        w_PE_to_P_mod = paras[1] * w_PE_to_P
        m_neuron_l, v_neuron_l, _ = run_mfn_circuit(w_PE_to_P_mod, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                        fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
    
        ### two columns
        
        ## generate stimuli
        np.random.seed(186)
        n_trials = np.int32(100)
        trial_duration_2 = np.int32(5000)
        num_values_per_trial_2 = np.int32(10)
        
        n_repeats_per_stim = dtype(trial_duration_2/num_values_per_trial_2)

        stimuli_2 = stimuli_moments_from_uniform(n_trials, num_values_per_trial_2, dtype(5 - np.sqrt(3)*5), 
                                               dtype(5 + np.sqrt(3)*5), dtype(0), dtype(5))
        stimuli_2 = np.repeat(stimuli_2, n_repeats_per_stim)
    
            
        ## run network - parameter smaller 
        w_PE_to_P_mod = paras[0] * dtype(w_PE_to_P)
        v_PE_to_P_mod = paras[0] * dtype(v_PE_to_P)
        [_, _, _, _, alpha_s, _, _] = run_mfn_circuit_coupled(w_PE_to_P_mod, w_P_to_PE, w_PE_to_PE, v_PE_to_P_mod, 
                                                              v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                              tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                              VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                              v_PE_to_V = v_PE_to_V) 
        
        ## run network - parameter larger 
        w_PE_to_P_mod = paras[1] * dtype(w_PE_to_P)
        v_PE_to_P_mod = paras[1] * dtype(v_PE_to_P)
        [_, _, _, _, alpha_l, _, _] = run_mfn_circuit_coupled(w_PE_to_P_mod, w_P_to_PE, w_PE_to_PE, v_PE_to_P_mod, 
                                                              v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                              tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                              VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                              v_PE_to_V = v_PE_to_V) 
    
        ## save
        with open(file_for_data,'wb') as f:
            pickle.dump([paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
                         m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l],f)
    
    else: # load results from previous simulation
        
        with open(file_for_data,'rb') as f:
            [paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
             m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
            
            
    ### plot single panels
    plot_example_mean(stimuli_1, trial_duration_1, m_neuron_s, figsize=(5,3), fs=7, lw=1.2)
    plot_example_variance(stimuli_1, trial_duration_1, v_neuron_s, figsize=(5,3), fs=7, lw=1.2)
    
    plot_example_mean(stimuli_1, trial_duration_1, m_neuron_l, figsize=(5,3), fs=7, lw=1.2)
    plot_example_variance(stimuli_1, trial_duration_1, v_neuron_l, figsize=(5,3), fs=7, lw=1.2)
    
    plt.figure()
    plt.plot(alpha_s, 'b')
    plt.plot(alpha_l, 'r')


# %% Robustness: wP2PE

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_change_wP2PE.pickle'
    
    ### get data
    if not plot_only: # simulate respective networks
    
        ## parameters tested
        paras = [0.5, 1.5]
    
        ## load default parameters
        VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
         v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
        
        ### one column
        
        ## generate stimuli
        np.random.seed(186)
        trial_duration_1 = np.int32(100000)
        num_values_per_trial_1 = np.int32(200)
        
        repeats_per_value = trial_duration_1//num_values_per_trial_1
        
        stimuli_1 = random_uniform_from_moments(5, 2, num_values_per_trial_1)
        stimuli_1 = np.repeat(stimuli_1, repeats_per_value)
        
        ## run network - parameter smaller 
        w_P_to_PE_mod = paras[0] * w_P_to_PE
        m_neuron_s, v_neuron_s, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE_mod, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                    fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
        
        ## run network - parameter larger
        w_P_to_PE_mod = paras[1] * w_P_to_PE
        m_neuron_l, v_neuron_l, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE_mod, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                        fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
    
        ### two columns
        
        ## generate stimuli
        np.random.seed(186)
        n_trials = np.int32(100)
        trial_duration_2 = np.int32(5000)
        num_values_per_trial_2 = np.int32(10)
        
        n_repeats_per_stim = dtype(trial_duration_2/num_values_per_trial_2)

        stimuli_2 = stimuli_moments_from_uniform(n_trials, num_values_per_trial_2, dtype(5 - np.sqrt(3)*5), 
                                               dtype(5 + np.sqrt(3)*5), dtype(0), dtype(5))
        stimuli_2 = np.repeat(stimuli_2, n_repeats_per_stim)
        
        ## run network - parameter smaller 
        w_P_to_PE_mod = paras[0] * dtype(w_P_to_PE)
        v_P_to_PE_mod = paras[0] * dtype(v_P_to_PE)
        [_, _, _, _, alpha_s, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE_mod, w_PE_to_PE, v_PE_to_P, 
                                                              v_P_to_PE_mod, v_PE_to_PE, tc_var_per_stim, 
                                                              tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                              VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                              v_PE_to_V = v_PE_to_V) 
        
        ## run network - parameter larger 
        w_P_to_PE_mod = paras[1] * dtype(w_P_to_PE)
        v_P_to_PE_mod = paras[1] * dtype(v_P_to_PE)
        [_, _, _, _, alpha_l, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE_mod, w_PE_to_PE, v_PE_to_P, 
                                                              v_P_to_PE_mod, v_PE_to_PE, tc_var_per_stim, 
                                                              tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                              VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                              v_PE_to_V = v_PE_to_V) 
    
        ## save
        with open(file_for_data,'wb') as f:
            pickle.dump([paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
                         m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l],f)
    
    else: # load results from previous simulation
        
        with open(file_for_data,'rb') as f:
            [paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
             m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
            
            
    ### plot single panels
    plot_example_mean(stimuli_1, trial_duration_1, m_neuron_s, figsize=(5,3), fs=7, lw=1.2)
    plot_example_variance(stimuli_1, trial_duration_1, v_neuron_s, figsize=(5,3), fs=7, lw=1.2)
    
    plot_example_mean(stimuli_1, trial_duration_1, m_neuron_l, figsize=(5,3), fs=7, lw=1.2)
    plot_example_variance(stimuli_1, trial_duration_1, v_neuron_l, figsize=(5,3), fs=7, lw=1.2)
    
    plt.figure()
    plt.plot(alpha_s, 'b')
    plt.plot(alpha_l, 'r')

    
# %% Robustness: wPE2V

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_change_wPE2V.pickle'
    
    ### get data
    if not plot_only: # simulate respective networks
    
        ## parameters tested
        paras = [0.5, 1.5]
    
        ## load default parameters
        VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
         v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
        
        ### one column
        
        ## generate stimuli
        np.random.seed(186)
        trial_duration_1 = np.int32(100000)
        num_values_per_trial_1 = np.int32(200)
        
        repeats_per_value = trial_duration_1//num_values_per_trial_1
        
        stimuli_1 = random_uniform_from_moments(5, 2, num_values_per_trial_1)
        stimuli_1 = np.repeat(stimuli_1, repeats_per_value)
        
        
        ## run network - parameter smaller 
        w_PE_to_V_mod = paras[0] * dtype(w_PE_to_V)
        m_neuron_s, v_neuron_s, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                    fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V_mod)
        
        ## run network - parameter larger
        w_PE_to_V_mod = paras[1] * dtype(w_PE_to_V)
        m_neuron_l, v_neuron_l, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                        fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V_mod)
        
        ### two columns
        
        ## generate stimuli
        np.random.seed(186)
        n_trials = np.int32(100)
        trial_duration_2 = np.int32(5000)
        num_values_per_trial_2 = np.int32(10)
        
        n_repeats_per_stim = dtype(trial_duration_2/num_values_per_trial_2)

        stimuli_2 = stimuli_moments_from_uniform(n_trials, num_values_per_trial_2, dtype(5 - np.sqrt(3)*5), 
                                               dtype(5 + np.sqrt(3)*5), dtype(0), dtype(5))
        stimuli_2 = np.repeat(stimuli_2, n_repeats_per_stim)
        
        ## run network - parameter smaller 
        w_PE_to_V_mod = paras[0] * dtype(w_PE_to_V)
        v_PE_to_V_mod = paras[0] * dtype(v_PE_to_V)
        [_, _, _, _, alpha_s, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                              v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                              tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                              VS = VS, VV = VV, w_PE_to_V = w_PE_to_V_mod, 
                                                              v_PE_to_V = v_PE_to_V_mod) 
        
        ## run network - parameter larger 
        w_PE_to_V_mod = paras[1] * dtype(w_PE_to_V)
        v_PE_to_V_mod = paras[1] * dtype(v_PE_to_V)
        [_, _, _, _, alpha_l, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                              v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                              tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                              VS = VS, VV = VV, w_PE_to_V = w_PE_to_V_mod, 
                                                              v_PE_to_V = v_PE_to_V_mod) 
    
        ## save
        with open(file_for_data,'wb') as f:
            pickle.dump([paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
                         m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l],f)
    
    else: # load results from previous simulation
        
        with open(file_for_data,'rb') as f:
            [paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
             m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
            
            
    ### plot single panels
    plot_example_mean(stimuli_1, trial_duration_1, m_neuron_s, figsize=(5,3), fs=7, lw=1.2)
    plot_example_variance(stimuli_1, trial_duration_1, v_neuron_s, figsize=(5,3), fs=7, lw=1.2)
    
    plot_example_mean(stimuli_1, trial_duration_1, m_neuron_l, figsize=(5,3), fs=7, lw=1.2)
    plot_example_variance(stimuli_1, trial_duration_1, v_neuron_l, figsize=(5,3), fs=7, lw=1.2)
    
    plt.figure()
    plt.plot(alpha_s, 'b')
    plt.plot(alpha_l, 'r')


# %% Robustness: additional top-down input

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_add_top_down.pickle'
    file_temp = '../results/data/moments/data_temp.pickle'
    
    ### get data
    if not plot_only: # simulate respective networks
    
        ## parameters tested
        paras = [0.5, 1]
        add_input = np.zeros(8)
        
        ### one column
        
        ### run network - add top-down 1
        add_input[2:4] = paras[0]
        add_input[5] = paras[0]
        add_input[7] = paras[0]
        [_, _, _, _, _, m_neuron_s, v_neuron_s] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_temp, add_input = add_input)
        
        ## run network - add top-down 2
        add_input[2:4] = paras[1]
        add_input[5] = paras[1]
        add_input[7] = paras[1]
        [_, _, _, _, _, m_neuron_l, v_neuron_l] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_temp, add_input = add_input)
        
        ### two columns
        
        ## run network - add top-down 1
        cell_ids = np.array([2,3,5,7])
        _, alpha_s = simulate_effect_baseline(mfn_flag, 1, 1, 0, [paras[0]], cell_ids, record_last_alpha = True)
        
        ## run network - ad top-down 2
        _, alpha_l = simulate_effect_baseline(mfn_flag, 1, 1, 0, [paras[1]], cell_ids, record_last_alpha = True)
        
        ## save
        with open(file_for_data,'wb') as f:
            pickle.dump([paras, alpha_s, alpha_l, m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l],f)
    
    else: # load results from previous simulation
        
        with open(file_for_data,'rb') as f:
            [paras, alpha_s, alpha_l, m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
            
            
    ### plot single panels
    # plot_example_mean(stimuli_1, trial_duration_1, m_neuron_s, figsize=(5,3), fs=7, lw=1.2)
    # plot_example_variance(stimuli_1, trial_duration_1, v_neuron_s, figsize=(5,3), fs=7, lw=1.2)
    
    # plot_example_mean(stimuli_1, trial_duration_1, m_neuron_l, figsize=(5,3), fs=7, lw=1.2)
    # plot_example_variance(stimuli_1, trial_duration_1, v_neuron_l, figsize=(5,3), fs=7, lw=1.2)
    
    plt.figure()
    plt.plot(m_neuron_s, 'b')
    plt.plot(m_neuron_l, 'r')
    
    plt.figure()
    plt.plot(v_neuron_s, 'b')
    plt.plot(v_neuron_l, 'r')
    
    plt.figure()
    plt.plot(alpha_s, 'b')
    plt.plot(alpha_l, 'r')
    

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
