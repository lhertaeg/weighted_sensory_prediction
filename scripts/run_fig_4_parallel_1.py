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
import os.path

from src.functions_simulate import simulate_neuromod, simulate_moment_estimation_upon_changes_PE, simulate_neuromod_effect_on_neuron_properties
from src.functions_simulate import simulate_neuromod_combos, simulate_neuromod_pert
from src.plot_data import plot_heatmap_neuromod, plot_combination_activation_INs, plot_neuromod_per_net, plot_points_of_interest_neuromod, plot_bar_neuromod
from src.plot_data import plot_illustration_changes_upon_baseline_PE, plot_illustration_changes_upon_gain_PE, plot_changes_upon_input2PE_neurons, plot_bar_neuromod_stacked
from src.plot_data import plot_neurmod_results, plot_neuromod_impact_inter, plot_changes_upon_input2PE_neurons_new, plot_influence_interneurons_baseline_or_gain

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Neuromodulator targeting one IN (with various strengths)

run_cell = True
plot_only = False
column = 0 

xp, xs, xv = 1, 0, 0

if run_cell:
    
    ### filename for data
    file_for_data = '../results/data/neuromod/data_neuromod_' + str(xp) + '_' + str(xs) + '_' + str(xv) + '.pickle'
    
    ### define perturbations tested
    pert_strength = np.linspace(0,0.5,6)
    
    ### main loop
    if not plot_only:
        
        alpha = np.zeros((len(pert_strength), 3, 2))
    
        for k, mfn_flag in enumerate(['10', '01', '11']):
            
            for std_mean in [0, 1]: # uncertainty of environement [0, 1]
                
                n_std = 1 - std_mean # uncertainty of stimulus [1, 0]
                print('MFN: ', mfn_flag, ', Input case', std_mean)
                
                [_, _, _, _, alpha_after_pert] = simulate_neuromod_pert(mfn_flag, pert_strength, std_mean, n_std, column, 
                                                                        xp, xs, xv, file_for_data = 'Test')
                
                alpha[:,k,std_mean] = alpha_after_pert
                    
                
        ### save data
        with open(file_for_data,'wb') as f:
            pickle.dump([xp, xs, xv, pert_strength, alpha],f) 
        
    else:
        
        ### load data
        with open(file_for_data,'rb') as f:
            [xp, xs, xv, pert_strength, alpha] = pickle.load(f)
            

    ### plot data    
    #plot_changes_upon_input2PE_neurons_new()
    plt.figure()
    plt.plot(pert_strength, alpha[:,:,0])
    
    plt.figure()
    plt.plot(pert_strength, alpha[:,:,1])
        

# %% Activate fractions of IN neurons in lower/higher PE circuit or both for a specified input statistics

run_cell = False

if run_cell:
    
    # define combo activation resolution
    nums = 11
    
    # run through all combinations
    for mfn_flag in ['01']: # choose mean-field network to simulate
        
        print('MFN code: ', mfn_flag)
        print(' ')
    
        for column in range(3): # 0: both, 1: lower level PE circuit, 2: higher level PE circuit

            for std_mean in [0, 1]: # uncertainty of environement [0, 1]
                
                n_std = 1 - std_mean # uncertainty of stimulus [1, 0]
                    
                for IN_combo_flg in range(3): # 0: PV - SOM, 1: SOM - VIP, 2: VIP - PV
        
                    # filename for data & define activation combos
                    identifier = '_column_' + str(column) + '_acrossvar_' + str(std_mean) + '_withinvar_' + str(n_std)
                    print(identifier)
        
                    if IN_combo_flg==0:
                        file_for_data = '../results/data/neuromod/data_weighting_neuromod_PV-SOM_' + mfn_flag + identifier + '.pickle'
                        xp = np.linspace(0, 1, nums)
                        xs = 1 - xp
                        xv = np.zeros_like(xp)
                        
                    elif IN_combo_flg==1:
                        file_for_data = '../results/data/neuromod/data_weighting_neuromod_SOM-VIP_' + mfn_flag + identifier + '.pickle'
                        xp = np.zeros(nums)
                        xs = np.linspace(0, 1, nums)
                        xv = 1 - xs
                        
                    elif IN_combo_flg==2:
                        file_for_data = '../results/data/neuromod/data_weighting_neuromod_VIP-PV_' + mfn_flag + identifier + '.pickle'
                        xv = np.linspace(0, 1, nums)
                        xp = 1 - xv
                        xs = np.zeros(nums)
            
        
                    if not os.path.exists(file_for_data):
                        [_, _, _, alpha_before_pert, 
                         alpha_after_pert] = simulate_neuromod_combos(mfn_flag, std_mean, n_std, column, 
                                                                      xp, xs, xv, file_for_data = file_for_data)
        

# %% Plot results above

run_cell = False

if run_cell:
    
    column = 0
    std_mean = 0
    n_std = 1
    
    plot_neuromod_impact_inter(column, std_mean, n_std, s=30)


# %% How are the variance neurons influenced by changes in nPE and pPE neurons in the lower and higher PE circuit?

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

        plot_changes_upon_input2PE_neurons_new()

    
# %% Test gain and BL of nPE and pPE neurons

run_cell = False
plot_only = True

if run_cell:
    
    marker = ['o', 's', 'd']
    
    ### choose mean-field network to simulate
    for i_net, mfn_flag in enumerate(['10', '01', '11']):
                                  
    #mfn_flag = '01' # valid options are '10', '01', '11
        file_for_data = '../results/data/neuromod/data_PE_props_vs_neuromod_' + mfn_flag + '.pickle'
    
        ### input statistics
        min_mean, max_mean, m_sd, n_sd = 5, 5, 0, np.sqrt(5)
        
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
    plot_influence_interneurons_baseline_or_gain(plot_annotation=False)
    plot_influence_interneurons_baseline_or_gain(plot_baseline=False, plot_annotation=False)
    