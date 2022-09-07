#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:10:26 2022

@author: loreen.hertaeg
"""


# %% Import

import numpy as np
import pickle

import sys
sys.path.append('../src')

from mean_field_model import default_para, stimuli_moments_from_uniform, run_mean_field_model, alpha_parameter_exploration
#from plot_toy_model import plot_limit_case, plot_alpha_para_exploration_ratios, plot_fraction_sensory_comparsion, plot_alpha_para_exploration
#from plot_toy_model import plot_manipulation_results
#from plot_results_mfn import plot_limit_case_example, plot_transitions_examples, heatmap_summary_transitions, plot_transition_course

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% Test: Exc/Inh perturbation, VIP in MFN1, +/- 2 perturbation, whole range

flag = 1

if flag==1:
    
    ### load and define parameters
    input_flg = '10'
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
    file_data4plot = file_data4plot = '../results/data/weighting_perturbation/test_weighting_perturbations_' + input_flg + '.pickle'
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)
    
    if input_flg=='10':
        nPE_scale = 1.015
        pPE_scale = 1.023
    elif input_flg=='01':
        nPE_scale = 1.7 # 1.72
        pPE_scale = 1.7 # 1.68
    elif input_flg=='11':
        nPE_scale = 2.49
        pPE_scale = 2.53
        
    w_PE_to_P[0,0] *= nPE_scale * 15 # !!!!!!!!!!! to make it faster
    w_PE_to_P[0,1] *= pPE_scale * 15 # !!!!!!!!!!!
    w_PE_to_V = [nPE_scale, pPE_scale]
    
    v_PE_to_P[0,0] *= nPE_scale * 0.7 # !!!!!!!!!!! to make it slower
    v_PE_to_P[0,1] *= pPE_scale * 0.7 # !!!!!!!!!!!
    v_PE_to_V = [nPE_scale, pPE_scale]
    
    tc_var_per_stim = dtype(1000)
    tc_var_pred = dtype(1000)
    
    ### stimulation & simulation parameters
    n_trials = np.int32(400)
    last_n = np.int32(100)
    trial_duration = np.int32(5000)# dtype(5000)
    n_stimuli_per_trial = np.int32(10)
    n_repeats_per_stim = np.int32(trial_duration/n_stimuli_per_trial)
    n_repeats = np.int32(1) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    ### perturbation specs
    perturbations = [-2, 2]
    
    ### means and std's to be tested
    mean_mean, min_std = dtype(3), dtype(0)
    std_mean_arr = np.linspace(0,3,5, dtype=dtype)
    std_std_arr = np.linspace(0,3,5, dtype=dtype)
    # std_mean_arr = np.array([3, 2.25, 1.5, 0.75, 0])
    # std_std_arr = np.array([0, 0.75, 1.5, 2.25, 3])
    
    ### initalise
    frac_sens_before_pert = np.zeros((len(perturbations), 4, len(std_std_arr),len(std_mean_arr)))
    frac_sens_after_pert = np.zeros((len(perturbations), 4, len(std_std_arr),len(std_mean_arr)))
    weighted_out = np.zeros((len(perturbations), 4, len(std_std_arr),len(std_mean_arr), np.int32(n_trials * trial_duration)))
    
    ### compute variances and predictions
    for id_mod, perturbation_strength in enumerate(perturbations): 
            
            for id_cell_perturbed in range(7,8): #range(4,8): # (nPE, pPE, nPE dend, pPE dend,) PVv, PVm, SOM, VIP
            
                ### display progress
                print(str(id_mod+1) + '/' + str(len(perturbations)) + ' and ' + str(id_cell_perturbed-3) + '/' + str(4))
        
                ## add perturbation
                perturbation = np.zeros((n_trials * trial_duration,8))                          
                perturbation[(n_trials * trial_duration)//2:, id_cell_perturbed] = perturbation_strength          
                fixed_input_plus_perturbation = fixed_input + perturbation
                
                for col, std_mean in enumerate(std_mean_arr):
                    for row, std_std in enumerate(std_std_arr):
                    
                        print(str(col+1) + '/' + str(len(std_mean_arr)) + ' and ' + str(row+1) + '/' + str(len(std_std_arr)))
                        
                        ### define stimuli 
                        np.random.seed(186)
                        
                        stimuli = stimuli_moments_from_uniform(n_trials, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                                               dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
                        
                        stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))
                        
                        
                        ### run model
                        [_, _, _, _, alpha, _, 
                         weighted_output] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                 tc_var_per_stim, tc_var_pred, tau_pe, fixed_input_plus_perturbation, stimuli)
                        
                        ###  check for inf and nan
                        if ((sum(np.isinf(alpha))>0) or (sum(np.isnan(alpha))>0)):
                            print('Warning: computation yields nan or inf in alpha.')
                        
                        ### fraction of sensory input in weighted output
                        frac_sens_before_pert[id_mod, id_cell_perturbed-4, row, col] = np.mean(alpha[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
                        frac_sens_after_pert[id_mod, id_cell_perturbed-4, row, col] = np.mean(alpha[(n_trials - last_n) * trial_duration:])
         
                        ### weihgted output
                        weighted_out[id_mod, id_cell_perturbed-4, row, col, :] = weighted_output
                    
    ### save data
    with open(file_data4plot,'wb') as f:
        pickle.dump([n_trials, last_n, trial_duration, 
                     frac_sens_before_pert, frac_sens_after_pert, weighted_out],f)  
     
       

# %% Exc/Inh perturbation, all IN neurons, all MFN

# run for each MFN network

flag = 0

if flag==1:
    
    for column in [1,2]:
    
        ### define MFN networks that will be simulated
        input_flgs = ['10', '01', '11']
        
        for input_flg in input_flgs:
        
            ### load and define parameters
            #input_flg = '10'
            filename = '../results/data/moments/Data_Optimal_Parameters_MFN_10.pickle'
            file_data4plot = '../results/data/weighting_perturbation/data_weighting_perturbations_' + input_flg + '_column_' + str(column) + '.pickle'
            
            [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
             tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename)
            
            if input_flg=='10':
                nPE_scale = 1.015
                pPE_scale = 1.023
            elif input_flg=='01':
                nPE_scale = 1.7 # 1.72
                pPE_scale = 1.7 # 1.68
            elif input_flg=='11':
                nPE_scale = 2.49
                pPE_scale = 2.53
                
            w_PE_to_P[0,0] *= nPE_scale * 15 # !!!!!!!!!!! to make it faster
            w_PE_to_P[0,1] *= pPE_scale * 15 # !!!!!!!!!!!
            w_PE_to_V = [nPE_scale, pPE_scale]
            
            v_PE_to_P[0,0] *= nPE_scale * 0.7 # !!!!!!!!!!! to make it slower
            v_PE_to_P[0,1] *= pPE_scale * 0.7 # !!!!!!!!!!!
            v_PE_to_V = [nPE_scale, pPE_scale]
            
            tc_var_per_stim = dtype(1000)
            tc_var_pred = dtype(1000)
            
            ### stimulation & simulation parameters
            n_trials = np.int32(400)
            last_n = np.int32(100)
            trial_duration = np.int32(5000)# dtype(5000)
            n_stimuli_per_trial = np.int32(10)
            n_repeats_per_stim = np.int32(trial_duration/n_stimuli_per_trial)
            n_repeats = np.int32(1) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            ### perturbation specs
            perturbations = [-1, 1]
            
            ### means and std's to be tested
            mean_mean, min_std = dtype(3), dtype(0)
            std_mean_arr = np.array([3, 2.25, 1.5, 0.75, 0])
            std_std_arr = np.array([0, 0.75, 1.5, 2.25, 3])
            
            ### initalise
            frac_sens_before_pert = np.zeros((len(perturbations), 4, len(std_mean_arr)))
            frac_sens_after_pert = np.zeros((len(perturbations), 4, len(std_mean_arr)))
            
            ### compute variances and predictions
            for id_mod, perturbation_strength in enumerate(perturbations): 
                    
                    for id_cell_perturbed in range(4,8): # (nPE, pPE, nPE dend, pPE dend,) PVv, PVm, SOM, VIP
                    
                        ### display progress
                        print(str(id_mod+1) + '/' + str(len(perturbations)) + ' and ' + str(id_cell_perturbed-3) + '/' + str(4))
                
                        ## add perturbation
                        perturbation = np.zeros((n_trials * trial_duration,8))                          
                        perturbation[(n_trials * trial_duration)//2:, id_cell_perturbed] = perturbation_strength 
                        fixed_input_plus_perturbation = fixed_input + perturbation
                        
                        if column==1:
                            fixed_input_1 = fixed_input_plus_perturbation
                            fixed_input_2 = fixed_input
                        else:
                            fixed_input_1 = fixed_input
                            fixed_input_2 = fixed_input_plus_perturbation
                            
                        
                        for id_stim in range(len(std_mean_arr)):
                            
                            print(str(id_stim+1) + '/' + str(len(std_mean_arr)))
                            
                            ### define stimuli
                            std_mean = std_mean_arr[id_stim]
                            std_std = std_std_arr[id_stim]
                            
                            np.random.seed(186)
                            
                            stimuli = stimuli_moments_from_uniform(n_trials, n_stimuli_per_trial, dtype(mean_mean - np.sqrt(3)*std_mean), 
                                                                   dtype(mean_mean + np.sqrt(3)*std_mean), dtype(min_std), dtype(min_std + 2*np.sqrt(3)*std_std))
                            
                            stimuli = dtype(np.repeat(stimuli, n_repeats_per_stim))
                            
                            
                            ### run model
                            [_, _, _, _, alpha, _, _] = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                          tc_var_per_stim, tc_var_pred, tau_pe, None, stimuli,
                                                                          fixed_input_1 = fixed_input_1, 
                                                                          fixed_input_2 = fixed_input_2)
                            
                            ###  check for inf and nan
                            if ((sum(np.isinf(alpha))>0) or (sum(np.isnan(alpha))>0)):
                                print('Warning: computation yields nan or inf in alpha.')
                            
                            ### fraction of sensory input in weighted output
                            frac_sens_before_pert[id_mod, id_cell_perturbed-4, id_stim] = np.mean(alpha[(n_trials//2 - last_n) * trial_duration:n_trials//2 * trial_duration])
                            frac_sens_after_pert[id_mod, id_cell_perturbed-4, id_stim] = np.mean(alpha[(n_trials - last_n) * trial_duration:])
             
                            
            ### save data
            with open(file_data4plot,'wb') as f:
                pickle.dump([n_trials, last_n, trial_duration, frac_sens_before_pert, frac_sens_after_pert],f)  
                                        
