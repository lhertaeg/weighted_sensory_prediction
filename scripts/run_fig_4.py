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
from src.plot_data import plot_illustration_changes_upon_baseline_PE, plot_illustration_changes_upon_gain_PE, plot_changes_upon_input2PE_neurons, plot_bar_neuromod_stacked
from src.plot_data import plot_neurmod_results

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Activate fractions of IN neurons in lower/higher PE circuit or both for a specified input statistics

run_cell = False
plot_only = True

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
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
    
    #f, ax = plt.subplots(1, 3, figsize=(15,3)) #  plt.subplots(2, 3, figsize=(12,6))

    #for flag_what in range(2): # 0: VIP-PV, 1: VIP-SOM
    #for column in range(3):
        #plot_bar_neuromod(column, mfn_flag, flag_what, dgrad = 0.001, axs=ax[flag_what,column])
        #plot_bar_neuromod_stacked(column, mfn_flag, dgrad = 0.001, axs=ax[column])
        
    plot_neurmod_results(0)
    plot_neurmod_results(1)
    

# %% How is the sensory weight influenced by the variance (illustrate)

run_cell = False

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
cmap_sensory_prediction = LinearSegmentedColormap.from_list(name='cmap_sensory_prediction', colors=['#19535F','#D6D6D6','#D76A03'])

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
    
    
# %% How are the v & m neurons influenced by changes in nPE and pPE neurons in the lower and higher PE circuit?

run_cell = True

if run_cell:
     
    plot_changes_upon_input2PE_neurons()
            

# %%  How is the variance influenced by BL of nPE and pPE in lower and higher PE circuit?    
 
run_cell = True

if run_cell:
    
    plot_illustration_changes_upon_baseline_PE()
            
            
# %%  How is the variance influenced by gain of nPE and pPE in lower and higher PE circuit?    
 
run_cell = True

if run_cell:
    
    plot_illustration_changes_upon_gain_PE()

    
# %% Test gain and BL of nPE and pPE neurons

run_cell = False
plot_only = True

if run_cell:
    
    f1, ax1 = plt.subplots(1,1)
    f2, ax2 = plt.subplots(1,1)
    marker = ['o', 's', 'd']
    
    ### choose mean-field network to simulate
    for i_net, mfn_flag in enumerate(['10', '01', '11']):
                                  
    #mfn_flag = '01' # valid options are '10', '01', '11
        file_for_data = '../results/data/neuromod/data_PE_props_vs_neuromod_' + mfn_flag + '.pickle'
    
        ### input statistics
        min_mean, max_mean, m_sd, n_sd = 5, 5, 0, 2.3
        
        if not plot_only:
            ### initiate
            results_base_nPE = np.zeros(4)
            results_base_pPE = np.zeros(4)
            results_gain_nPE = np.zeros(4)
            results_gain_pPE = np.zeros(4)
            
            ### run network without perturbation
            print('Without neuromodulation')
            [baseline_nPE, baseline_pPE, 
             gain_nPE, gain_pPE] = simulate_neuromod_effect_on_neuron_properties(mfn_flag, min_mean, max_mean, m_sd, n_sd)
            
            results_base_nPE[0] = baseline_nPE
            results_base_pPE[0] = baseline_pPE
            results_gain_nPE[0] = gain_nPE
            results_gain_pPE[0] = gain_pPE
            
            ### run network for which IN neurons are additionally stimulated
            print('With neuromodulation')
            list_id_cells = [[4,5], 6, 7]
            
            for k, id_cell in enumerate(list_id_cells):
            
                print(id_cell)
                
                [baseline_nPE, baseline_pPE, 
                 gain_nPE, gain_pPE] = simulate_neuromod_effect_on_neuron_properties(mfn_flag, min_mean, max_mean, m_sd, n_sd, id_cell=id_cell)
            
                results_base_nPE[k+1] = baseline_nPE
                results_base_pPE[k+1] = baseline_pPE
                results_gain_nPE[k+1] = gain_nPE
                results_gain_pPE[k+1] = gain_pPE
                
            ### save data
            with open(file_for_data,'wb') as f:
                pickle.dump([results_base_nPE, results_base_pPE, results_gain_nPE, results_gain_pPE],f) 
                
        else:
            
            ### load data
            with open(file_for_data,'rb') as f:
                [results_base_nPE, results_base_pPE, results_gain_nPE, results_gain_pPE] = pickle.load(f) 
            
            
        ### plot
        ax1.scatter(results_base_nPE[1:] - results_base_nPE[0], results_base_pPE[1:] - results_base_pPE[0], c=np.arange(3), marker=marker[i_net])
        ax2.scatter(results_gain_nPE[1:] - results_gain_nPE[0], results_gain_pPE[1:] - results_gain_pPE[0], c=np.arange(3), marker=marker[i_net])
        
    
    ax1.axline((0,0),slope=-1, color='k', ls=':')
    ax1.axhline(0, color='k', ls='--', alpha=0.5, zorder=0)
    ax1.axvline(0, color='k', ls='--', alpha=0.5, zorder=0)
    ax1.set_xlabel('BL nPE')
    ax1.set_ylabel('BL pPE')
    #ax.spines['bottom'].set_position('zero')
    #ax.spines['left'].set_position('zero')
    xbound = max(np.abs(ax1.get_xlim()))
    ybound = max(np.abs(ax1.get_ylim()))
    ax1.set_xlim([-xbound,xbound])
    ax1.set_ylim([-ybound,ybound])
    #ax1.axhspan(-ybound, 0, xmin=0, xmax=0.5, alpha=0.1)
    #ax1.axhspan(0, ybound, xmin=0.5, xmax=1, alpha=0.1, color='r')
    ax1.fill_between(np.linspace(-xbound,xbound), -np.linspace(-xbound,xbound), -ybound, alpha=0.1)
    ax1.fill_between(np.linspace(-xbound,xbound), ybound, -np.linspace(-xbound,xbound), color='r', alpha=0.1)
    
    ax2.axline((0,0),slope=-1, color='k', ls=':')
    ax2.axhline(0, color='k', ls='--', alpha=0.5, zorder=0)
    ax2.axvline(0, color='k', ls='--', alpha=0.5, zorder=0)
    ax2.set_xlabel(r'$\Delta$ gain nPE')
    ax2.set_ylabel(r'$\Delta$ gain pPE')
    #ax.spines['bottom'].set_position('zero')
    #ax.spines['left'].set_position('zero')
    
    xbound = max(np.abs(ax2.get_xlim()))
    ybound = max(np.abs(ax2.get_ylim()))
    ax2.set_xlim([-xbound,xbound])
    ax2.set_ylim([-ybound,ybound])
    #ax2.axhspan(-ybound, 0, xmin=0, xmax=0.5, alpha=0.1)
    #ax2.axhspan(0, ybound, xmin=0.5, xmax=1, alpha=0.1, color='r')
    ax2.fill_between(np.linspace(-xbound,xbound), -np.linspace(-xbound,xbound), -ybound, alpha=0.1)
    ax2.fill_between(np.linspace(-xbound,xbound), ybound, -np.linspace(-xbound,xbound), color='r', alpha=0.1)
     
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    
  
# %% plot for all MFN together

# Please note that diagonal is just an illustration of the boundary
# it actually depends on the mean and the std of the inputs (input statistics)
# also, it is only true in the purely linear case (not taking rectifications into account)

run_cell = False

if run_cell:
    
    ### load data
    file_for_data = '../results/data/neuromod/data_PE_props_vs_neuromod_10.pickle'
    with open(file_for_data,'rb') as f:
            [results_base_nPE_10, results_base_pPE_10, results_gain_nPE_10, results_gain_pPE_10] = pickle.load(f) 
            
    file_for_data = '../results/data/neuromod/data_PE_props_vs_neuromod_11.pickle'
    with open(file_for_data,'rb') as f:
            [results_base_nPE_11, results_base_pPE_11, results_gain_nPE_11, results_gain_pPE_11] = pickle.load(f) 
            
    file_for_data = '../results/data/neuromod/data_PE_props_vs_neuromod_01.pickle'
    with open(file_for_data,'rb') as f:
            [results_base_nPE_01, results_base_pPE_01, results_gain_nPE_01, results_gain_pPE_01] = pickle.load(f)
            
    ### plot data
    label_text = ['10','11','01']
    
    def create_legend_for_INS(c):
        
        if c==0:
            res = 'PV'
        elif c==1:
            res = 'SOM'
        elif c==2:
            res = 'VIP'
            
        return res
    
    plt.figure()
    
    sc = plt.scatter((results_base_nPE_10[1:] - results_base_nPE_10[0]) + (results_base_pPE_10[1:] - results_base_pPE_10[0]),
                (results_gain_nPE_10[1:] - results_gain_nPE_10[0]) + (results_gain_pPE_10[1:] - results_gain_pPE_10[0]), c=np.arange(3), marker='o')
    
    plt.scatter((results_base_nPE_11[1:] - results_base_nPE_11[0]) + (results_base_pPE_11[1:] - results_base_pPE_11[0]),
                (results_gain_nPE_11[1:] - results_gain_nPE_11[0]) + (results_gain_pPE_11[1:] - results_gain_pPE_11[0]), c=np.arange(3), marker='s')
    
    plt.scatter((results_base_nPE_01[1:] - results_base_nPE_01[0]) + (results_base_pPE_01[1:] - results_base_pPE_01[0]),
                (results_gain_nPE_01[1:] - results_gain_nPE_01[0]) + (results_gain_pPE_01[1:] - results_gain_pPE_01[0]), c=np.arange(3), marker='d')
    
    ax = plt.gca()
    
    
    handles = sc.legend_elements()[0]
    legend1 = ax.legend(title='Targets', handles=handles, labels=['PV','SOM','VIP'], frameon=True)
    ax.add_artist(legend1)
    
    xbound = max(np.abs(ax.get_xlim()))
    ybound = max(np.abs(ax.get_ylim()))
    ax.set_xlim([-xbound,xbound])
    ax.set_ylim([-ybound,ybound])
    ax.axhspan(-ybound, 0, xmin=0, xmax=0.5, alpha=0.1)
    ax.axhspan(0, ybound, xmin=0.5, xmax=1, alpha=0.1, color='r')
    
    ax.set_xlabel('BL (nPE + pPE)')#, loc='left', labelpad=120)
    ax.set_ylabel(r'$\Delta$ gain (nPE + pPE)')#, loc='bottom', labelpad=150)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    plt.scatter(np.nan, np.nan, c='k', marker='o', label='10')
    plt.scatter(np.nan, np.nan, c='k', marker='s', label='11')
    plt.scatter(np.nan, np.nan, c='k', marker='d', label='01')
    legend2 = ax.legend(loc="upper right", title="MFN", frameon=True)
    ax.add_artist(legend2)
    
    ax.text(0.3,0.4,'Variance \nincreases', color='r')
    ax.text(-0.6,-0.4,'Variance \ndecreases', color='b')
    
    # ax.axline((0,0),slope=-1, color='k', ls=':') think about boundary, as it depends on input stats nt sure if it makes sense to show a line!?
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    #ax.spines['bottom'].set_position('zero')
    #ax.spines['left'].set_position('zero')
    sns.despine(ax=ax)
    
    