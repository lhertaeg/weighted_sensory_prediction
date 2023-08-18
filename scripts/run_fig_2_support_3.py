#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:50:33 2023

@author: loreen.hertaeg
"""

# %% Import

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.default_parameters import Neurons, Activity_Zero, Network, Stimulation
from src.functions_save import load_network_para
from src.functions_networks import run_population_net
from src.functions_simulate import simulate_spatial_example
from src.plot_data import plot_examples_spatial_M, plot_examples_spatial_V, plot_deviation_spatial

dtype = np.float32



# %% Test several input statistcis

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_spatial_mfn_diff_input_statistics.pickle'
    file_temporary = '../results/data/moments/data_spatial_mfn_temporary.pickle'
    
    ### input statistic parameterisation
    means_tested = np.linspace(3,6,7)
    stds_tested = np.linspace(np.sqrt(4),np.sqrt(8), 5)
    
    ### initialise
    deviation_mean = np.zeros((len(means_tested), len(stds_tested)))
    deviation_std = np.zeros((len(means_tested), len(stds_tested)))
    
    ### get data
    if not plot_only: # simulate respective network
        
        for i, mean in enumerate(means_tested):
            for j, std in enumerate(stds_tested):
        
                print('Mean no.: ' , (i+1) , '/' , len(means_tested), ', Std no.: ', (j+1), '/', len(stds_tested))
                [mean_stimulus, spatial_std, spatial_noise, 
                 num_time_steps, m_neuron, v_neuron, rates_final] = simulate_spatial_example(mfn_flag, dtype(mean), dtype(std), file_temporary, 
                                                                                num_sub_nets = np.int32(1000), num_time_steps = np.int32(4000))
                                                                                             
                deviation_mean[i,j] = (m_neuron[-1] - mean) / mean
                deviation_std[i,j] = (v_neuron[-1] - std**2) / std**2
                
        ### save data
        with open(file_for_data,'wb') as f:
            pickle.dump([means_tested, stds_tested, deviation_mean, deviation_std],f)
                                                                                        
    else: # load results

        with open(file_for_data,'rb') as f:
            [means_tested, stds_tested, deviation_mean, deviation_std] = pickle.load(f)

    ### plot data
    
    plot_deviation_spatial(deviation_mean, means_tested, stds_tested, vmin=0, vmax=5, fs=6, ax=None)
    plot_deviation_spatial(deviation_std, means_tested, stds_tested, vmin=0, vmax=5, fs=6, show_mean=False, ax=None)


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
         num_time_steps, m_neuron, v_neuron, rates_final] = simulate_spatial_example(mfn_flag, dtype(8), dtype(2), file_for_data, 
                                                                        num_sub_nets = np.int32(1000), num_time_steps = np.int32(4000),
                                                                        M_init = m_neuron_before[-1], V_init = v_neuron_before[-1], 
                                                                        rates_init = rates_final_before)#, dist_type='normal')
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron, v_neuron, rates_final] = pickle.load(f)


    plot_examples_spatial_M(num_time_steps, m_neuron_before, m_neuron, 5, 8, labels=['Before','After'])
    plot_examples_spatial_V(num_time_steps, v_neuron_before, v_neuron, 2**2, 2**2, labels=['Before','After'])


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
         num_time_steps, m_neuron, v_neuron, rates_final] = simulate_spatial_example(mfn_flag, dtype(5), dtype(np.sqrt(6)), file_for_data, 
                                                                        num_sub_nets = np.int32(1000), num_time_steps = np.int32(4000),
                                                                        M_init = m_neuron_before[-1], V_init = v_neuron_before[-1], 
                                                                        rates_init = rates_final_before)#, dist_type='normal')
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron, v_neuron, rates_final] = pickle.load(f)


    plot_examples_spatial_M(num_time_steps, m_neuron_before, m_neuron, 5, 5, labels=['Before','After'])
    plot_examples_spatial_V(num_time_steps, v_neuron_before, v_neuron, 2**2, 6, labels=['Before','After'])


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
         num_time_steps, m_neuron, v_neuron, rates_final] = simulate_spatial_example(mfn_flag, dtype(5), dtype(2), file_for_data, 
                                                                                     num_sub_nets = np.int32(1000), 
                                                                                     num_time_steps = np.int32(4000))#, dist_type='normal')
    
    else: # load results from previous simulation

        with open(file_for_data,'rb') as f:
            [mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron, v_neuron, rates_final] = pickle.load(f)


    plt.figure()
    plt.plot(np.arange(num_time_steps), m_neuron)
    ax = plt.gca()
    ax.axhline(mean_stimulus)
    
    plt.figure()
    plt.plot(np.arange(num_time_steps), v_neuron)
    ax = plt.gca()
    ax.axhline(spatial_std**2)
    

# %% Test whether the population model (on average, can also account for spatial changes)

# I highly doubt that, but oh well, let's try :)
# ... as expected ... it does not work => unconnected MFN necessary

run_cell = False

if run_cell:
    
    # get data
    with open('../results/data/population/data_population_network_parameters.pickle','rb') as f:
        [neurons_visual, inp_ext_soma, inp_ext_dend, weight_name, Dict_w, Dict_t] = pickle.load(f)
    
    with open('../results/data/population/data_population_network_para4corr_gain.pickle','rb') as f:
        [_, _, gain_factors_nPE_5, gain_factors_pPE_5, nPE_true, pPE_true] = pickle.load(f)
        
    # parametrisation neurons, network and rates
    NeuPar = Neurons()
    RatePar = Activity_Zero(NeuPar)
    NetPar = Network(NeuPar, Dict_w, Dict_t, weight_name, neurons_visual, gain_factors_nPE = gain_factors_nPE_5, 
                     gain_factors_pPE = gain_factors_pPE_5, nPE_true = nPE_true, pPE_true = pPE_true)
    
    StimPar = Stimulation(5, 0, inp_ext_soma, inp_ext_dend, neurons_visual, trial_duration = np.int32(20000), 
                          num_values_per_trial = np.int32(1)) # 20000, 20
    
    # run network
    run_population_net(NeuPar, NetPar, StimPar, RatePar, 0.1, folder='population', fln='spatial_test', std_spatial=2)
    
    
    # load simulation data
    PathData = '../results/data/population'
    arr = np.loadtxt(PathData + '/Data_PopulationNetwork_spatial_test.dat',delimiter=' ')
    t, R = arr[:,0], arr[:, 1:]
    
    ind_break = np.cumsum(NeuPar.NCells,dtype=np.int32)
    ind_break = np.concatenate([ind_break, np.array([340,341])])
    
    rE, rP, rS, rV, rD, r_mem, r_var = np.split(R, ind_break, axis=1)
    
    # plot data
    _, axs = plt.subplots(2,3, tight_layout=True, sharex=True)
    
    axs[0,0].plot(t, rE)
    axs[0,0].set_title('E')
    
    axs[0,1].plot(t, rD)
    axs[0,1].set_title('D')
    
    axs[0,2].plot(t, r_mem, 'b')
    axs[0,2].plot(t, r_var, 'r')
    axs[0,2].set_title('M (blue) & V(red)')
    
    axs[1,0].plot(t, rP)
    axs[1,0].set_title('P')
    
    axs[1,1].plot(t, rS)
    axs[1,1].set_title('S')
    
    axs[1,2].plot(t, rV)
    axs[1,2].set_title('V')