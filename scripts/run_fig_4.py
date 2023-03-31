#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.functions_simulate import simulate_neuromod, simulate_moment_estimation_upon_changes_PE, simulate_neuromod_effect_on_neuron_properties
from src.plot_data import plot_heatmap_neuromod, plot_combination_activation_INs, plot_neuromod_per_net, plot_points_of_interest_neuromod, plot_bar_neuromod
from src.plot_data import plot_illustration_changes_upon_baseline_PE, plot_illustration_changes_upon_gain_PE, plot_changes_upon_input2PE_neurons

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Activate fractions of IN neurons in lower/higher PE circuit or both for a specified input statistics

run_cell = False
plot_only = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '01' # valid options are '10', '01', '11
    
    ### define target area and stimulus statistics
    column = 0      # 0: both, 1: lower level PE circuit, 2: higher level PE circuit
    std_mean = 1   # uncertainty of environement [0, 0.5, 1]
    n_std = 0       # uncertainty of stimulus [1, 0.5, 0]
    
    ### filename for data
    identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
    file_for_data = '../results/data/neuromod/data_weighting_neuromod_' + mfn_flag + identifier + '.pickle'
    
    ### get data
    if not plot_only: # simulate respective network

        nums = 11
        xp, xs = dtype(np.meshgrid(np.linspace(0, 1, nums), np.linspace(0, 1, nums)))
        xv = np.sqrt(1 - xp**2 - xs**2)
        
        [_, _, _, alpha_before_pert, alpha_after_pert] = simulate_neuromod(mfn_flag, std_mean, n_std, column, 
                                                                            xp, xs, xv, file_for_data = file_for_data)
        
    else:
        
        with open(file_for_data,'rb') as f:
            [xp, xs, xv, alpha_before_pert, alpha_after_pert] = pickle.load(f)
            
     
    ### plot data
    plot_heatmap_neuromod(xp, xs, alpha_before_pert, alpha_after_pert)
    plot_combination_activation_INs(xp, xs, xv, alpha_before_pert, alpha_after_pert)
    
    
# %% Plot all heatmaps together

run_cell = False

if run_cell:
    
    mfn_flag = '01' # valid options are '10', '01', '11
    
    columns = [1,0,2]
    std_means = [1, 0.5, 0]
    n_std_all = [0, 0.5, 1]
    
    plot_neuromod_per_net(mfn_flag, columns, std_means, n_std_all)
    
    
# %% Plot points of interests in IN space 

# There is something odd, it doesn't seem to refelect what I see in heatmaps for 11!!!!

run_cell = False

if run_cell:
    
    mfn_flag = '11' # valid options are '10', '01', '11
    
    columns = [1,0,2]
    std_means = [1, 0.5, 0]
    n_std_all = [0, 0.5, 1]
    
    plot_points_of_interest_neuromod(mfn_flag, columns, std_means, n_std_all)
    plot_points_of_interest_neuromod(mfn_flag, columns, std_means, n_std_all, show_inter=True)
    
    
# %% Plot neuromodulation results in bar plots for VIP-PV and VIP-SOM

run_cell = False
mfn_flag = '10'

if run_cell:
    
    f, ax = plt.subplots(2, 3, figsize=(12,6))

    for flag_what in range(2): # 0: VIP-PV, 1: VIP-SOM
        for column in range(3):
            plot_bar_neuromod(column, mfn_flag, flag_what, dgrad = 0.001, axs=ax[flag_what,column])
    

# %% XXX

# sensory weight -- variances
# variance neuron -- nPE, pPE neuron
# nPE, pPE neurons -- INs

# %% How is the sensory weight influenced by the variance (illustrate)

run_cell = False

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
cmap_sensory_prediction = LinearSegmentedColormap.from_list(name='cmap_sensory_prediction', colors=['#19535F','#fefee3','#D76A03'])

if run_cell:
    
    sigma2_sens_fix, sigma2_pred_fix = 1, 1
    
    values_tested = np.linspace(-0.9,0.9,101)
    sigma2_delta_sens, sigma2_delta_pred = np.meshgrid(values_tested, values_tested)
    
    sensory_weight = (1/(sigma2_sens_fix + sigma2_delta_sens)) / (1/(sigma2_pred_fix + sigma2_delta_pred) + 1/(sigma2_sens_fix + sigma2_delta_sens))
    
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # #ax.contour3D(sigma2_delta_sens, sigma2_delta_pred, sensory_weight)
    # #ax.plot_wireframe(sigma2_delta_sens, sigma2_delta_pred, sensory_weight, cmap='flare')
    # ax.plot_surface(sigma2_delta_sens, sigma2_delta_pred, sensory_weight, rstride=1, cstride=1, cmap=cmap_sensory_prediction, edgecolor='none')
    
    # ax.set_xlabel('change in sensory var')
    # ax.set_ylabel('change in pre var')
    # ax.set_zlabel('sensory weight')
    
    plt.figure()
    data = pd.DataFrame(sensory_weight, index=np.round(values_tested,1), columns=np.round(values_tested,1))
    ax = sns.heatmap(data, cmap=cmap_sensory_prediction, vmin=0, vmax=1, xticklabels=20, yticklabels=20)
    ax.invert_yaxis()
    
    ax.set_xlabel('change in sensory var')
    ax.set_ylabel('change in pre var')
                
    # reduction in V neuron of first column leads to more sensory driven, while increase leads to more prediction driven
    # reduction in V neuron of second column leads to more prediction driven, while increase leads to more sensory driven
    # diagonal (same change in both columns, additive) remains 0.5 but deviations from it to the left/right are more blurred (fading out) when changes are postive
   
    
# %%  How is the variance influenced by BL of nPE and pPE in lower and higher PE circuit?    
 
run_cell = False

if run_cell:
    
    plot_illustration_changes_upon_baseline_PE()
            
            
# %%  How is the variance influenced by gain of nPE and pPE in lower and higher PE circuit?    
 
run_cell = False

if run_cell:
    
    plot_illustration_changes_upon_gain_PE()
    
    
# %% How are the variance neurons influenced by changes in nPE and pPE neurons in the lower and higher PE circuit?

# changes in nPE and pPE neurons in lower PE circuit will lead to changes in the variance neuron in the lower PE circuit 
# AND the higher PE circuit
# the latter changes occur due to changes in the prediction neuron of the lower PE circuit (see calculations & illustration)
# changes in PE neurons in higher PE circuit will lead to changes in the variance of the higher column
# changes in both will be superimposed of the first two ...

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### define target area and stimulus statistics
    column = 2      # 1: lower level PE circuit, 2: higher level PE circuit
    std_mean = 1   
    n_std = 1       
    
    ### filename for data
    identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
    file_for_data = '../results/data/neuromod/data_moments_vs_PE_neurons_' + mfn_flag + identifier + '.pickle'
    
    ### get data
    if not plot_only: # simulate respective network

        nums = 11
        pert_strength = np.linspace(-1,1,9)
        
        [pert_strength, m_act_lower, v_act_lower, v_act_higher] = simulate_moment_estimation_upon_changes_PE(mfn_flag, std_mean, n_std, column, pert_strength, 
                                                                                                            file_for_data = file_for_data)
        
    else:
        
        with open(file_for_data,'rb') as f:
            [pert_strength, m_act_lower, v_act_lower, v_act_higher] = pickle.load(f)
            
    ### plot data    
    f, axs = plt.subplots(1, 2, sharex=True, tight_layout=True)
    
    for i in range(2):
        # axs[i].plot(pert_strength, (m_act_lower[:,i] - m_act_lower[len(m_act_lower)//2,i]) / m_act_lower[len(m_act_lower)//2,i], color='b', ls=':')
        # axs[i].plot(pert_strength, (v_act_lower[:,i] - v_act_lower[len(v_act_lower)//2,i]) / v_act_lower[len(v_act_lower)//2,i], color='r', ls=':')
        # axs[i].plot(pert_strength, (v_act_higher[:,i] - v_act_higher[len(v_act_higher)//2,i]) / v_act_higher[len(v_act_higher)//2,i], color='r', ls='-')
    
        axs[i].plot(pert_strength, m_act_lower[:,i], color='b', ls=':')
        axs[i].plot(pert_strength, v_act_lower[:,i], color='r', ls=':')
        axs[i].plot(pert_strength, v_act_higher[:,i], color='r', ls='-')


# %% How are the v & m neurons influenced by changes in nPE and pPE neurons in the lower and higher PE circuit?

run_cell = False

if run_cell:
     
    plot_changes_upon_input2PE_neurons()
            
    
# %% Test gain and BL of nPE and pPE neurons

run_cell = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### input statistics
    min_mean, max_mean, m_sd, n_sd = 5, 5, 0, 2.3
    
    ### run network and extract 
    [baseline_nPE, baseline_pPE, 
     gain_nPE, gain_pPE] = simulate_neuromod_effect_on_neuron_properties(mfn_flag, min_mean, max_mean, m_sd, n_sd)
    
    
    