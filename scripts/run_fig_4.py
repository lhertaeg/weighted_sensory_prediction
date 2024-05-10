#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_moment_estimation_upon_changes_PE, simulate_neuromod_pert
from src.functions_simulate import simulate_neuromod_effect_on_neuron_properties
from src.plot_data import plot_changes_upon_input2PE_neurons_new, plot_influence_interneurons_baseline_or_gain, plot_neuromod_impact

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Neuromodulator targeting one IN (with various strengths)

run_cell = False
plot_only = False
column = 0 

xp, xs, xv = 1, 0, 0

if run_cell:
    
    ### filename for data
    file_for_data = '../results/data/neuromod/data_neuromod_' + str(xp) + '_' + str(xs) + '_' + str(xv) + '.pickle'
    
    ### define perturbations tested
    pert_strength = np.linspace(0,1,6)
    
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
    plot_neuromod_impact(pert_strength, alpha, xp, xs, xv)
                                
    
# %% Test gain and BL of nPE and pPE neurons

run_cell = True
plot_only = False

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
    