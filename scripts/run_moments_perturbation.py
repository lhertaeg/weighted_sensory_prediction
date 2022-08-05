#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 08:58:09 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.mean_field_model import default_para, stimuli_moments_from_uniform, run_mean_field_model, alpha_parameter_exploration
from src.mean_field_model import run_mean_field_model_one_column

from src.toy_model import random_uniform_from_moments, random_lognormal_from_moments, random_gamma_from_moments
from src.plot_toy_model import plot_limit_case, plot_alpha_para_exploration_ratios, plot_fraction_sensory_comparsion, plot_alpha_para_exploration
from src.plot_toy_model import plot_manipulation_results

from src.plot_results_mfn import plot_prediction, plot_variance, plot_mse, plot_manipulations, plot_deviations_upon_perturbations
from src.plot_results_mfn import plot_heatmap_perturbation_all

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Perturbing the E/I balance in PE circuits - example

flag = 0
flg_plot_only = 1

if flag==1:
    
    ### load and define parameters
    input_flg = '10' # 10, 01, 11
    
    VS, VV = int(input_flg[0]), int(input_flg[1])
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/moments/data_perturbation_example_' + input_flg + '.pickle'
    
    if flg_plot_only==0:
    
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename, VS=VS, VV=VV)
    
        if input_flg=='10':
            nPE_scale = 1.015
            pPE_scale = 1.023
        elif input_flg=='01':
            nPE_scale = 1.7 # 1.72
            pPE_scale = 1.7 # 1.68
        elif input_flg=='11':
            nPE_scale = 2.49
            pPE_scale = 2.53
            
        w_PE_to_P[0,0] *= nPE_scale
        w_PE_to_P[0,1] *= pPE_scale
        w_PE_to_V = [nPE_scale, pPE_scale]
        
        ### define stimuli
        n_trials = 1000
        trial_duration = 500
        n_stimuli_per_trial = 1
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        
        mean_stimuli, std_stimuli = 5, 2
        stimuli = random_uniform_from_moments(mean_stimuli, std_stimuli, (n_trials * n_stimuli_per_trial))
        stimuli = np.repeat(stimuli, n_repeats_per_stim)
        
        ### add perturbation
        id_cell_perturbed = 0
        perturbation_strength = -1
        fixed_input = np.tile(fixed_input, (len(stimuli),1))
        perturbation = np.zeros((len(stimuli),8))                          
        perturbation[(n_trials * trial_duration)//2:, id_cell_perturbed] = perturbation_strength          
        fixed_input_plus_perturbation = fixed_input + perturbation
        
        ### compute variances and predictions
        
        ## run model
        prediction, variance, PE_activity = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                            tc_var_per_stim, tau_pe, fixed_input_plus_perturbation, 
                                                                            stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
    
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([n_trials, trial_duration, stimuli, prediction, variance],f)                                      
     
    else:
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [n_trials, trial_duration, stimuli, prediction, variance] = pickle.load(f)  
        
        
    ### plot prediction and variance (in comparison to the data) 
    plot_prediction(n_trials, stimuli, trial_duration, prediction, perturbation_time=500, ylim_mse=[0,5])  
    plot_variance(n_trials, stimuli, trial_duration, variance, perturbation_time=500) 


# %% Perturbing the E/I balance in PE circuits 

flag = 0
flg_plot_only = 1

if flag==1:
    
    ### load and define parameters
    input_flg = '11' # 10, 01, 11
    
    VS, VV = int(input_flg[0]), int(input_flg[1])
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/moments/data_perturbations_' + input_flg + '.pickle'
    
    if flg_plot_only==0:
    
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename, VS=VS, VV=VV)
    
        if input_flg=='10':
            nPE_scale = 1.015
            pPE_scale = 1.023
        elif input_flg=='01':
            nPE_scale = 1.7 # 1.72
            pPE_scale = 1.7 # 1.68
        elif input_flg=='11':
            nPE_scale = 2.49
            pPE_scale = 2.53
            
        w_PE_to_P[0,0] *= nPE_scale
        w_PE_to_P[0,1] *= pPE_scale
        w_PE_to_V = [nPE_scale, pPE_scale]
        
        ### define stimuli
        n_trials = 1000
        min_trial_for_avg = 750
        trial_duration = 500
        n_stimuli_per_trial = 1
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        
        mean_stimuli, std_stimuli = 5, 2
        stimuli = random_uniform_from_moments(mean_stimuli, std_stimuli, (n_trials * n_stimuli_per_trial))
        stimuli = np.repeat(stimuli, n_repeats_per_stim)
        perturbations = [-1, 1]
    
        ### initialise
        dev_prediction_steady = np.zeros((len(perturbations), 8))
        dev_variance_steady = np.zeros((len(perturbations), 8))
        fixed_input = np.tile(fixed_input, (len(stimuli),1))
        
        ### compute variances and predictions
        for id_mod, perturbation_strength in enumerate(perturbations): 
            
            for id_cell_perturbed in range(8): # nPE, pPE, nPE dend, pPE dend, PVv, PVm, SOM, VIP
            
                ### display progress
                print(str(id_mod+1) + '/' + str(len(perturbations)) + ' and ' + str(id_cell_perturbed+1) + '/' + str(8))
        
                ## add perturbation
                perturbation = np.zeros((len(stimuli),8))                          
                perturbation[(n_trials * trial_duration)//2:, id_cell_perturbed] = perturbation_strength          
                fixed_input_plus_perturbation = fixed_input + perturbation
                
                ## run model
                prediction, variance, PE_activity = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                                    tc_var_per_stim, tau_pe, fixed_input_plus_perturbation, 
                                                                                    stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
                
                ## compute mean squared error
                running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
                dev_prediction = (prediction - running_average) / running_average
                
                mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
                momentary_variance = (stimuli - mean_running)**2
                variance_running = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1)
                dev_variance = (variance - variance_running) / variance_running
                
                dev_prediction_steady[id_mod, id_cell_perturbed] = np.mean(dev_prediction[(min_trial_for_avg * trial_duration):]) * 100
                dev_variance_steady[id_mod, id_cell_perturbed] = np.mean(dev_variance[(min_trial_for_avg * trial_duration):]) * 100
        
            
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([n_trials, min_trial_for_avg, trial_duration, dev_prediction_steady, dev_variance_steady],f)                                    
     
    else:

        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [n_trials, min_trial_for_avg, trial_duration, dev_prediction_steady, dev_variance_steady] = pickle.load(f)
        
        
    ### plot prediction and variance (in comparison to the data) 
    xticklabels = ['nPE', 'pPE', 'nPE dend', 'pPE dend', 'PVv', 'PVm', 'SOM', 'VIP']
    plot_manipulations(dev_prediction_steady, xticklabels, 'Prediction')
    plot_manipulations(dev_variance_steady, xticklabels, 'Variance')#, ylim=[-100,100])
    

# %% Plot deviations for all perturbations for one MFN

flag = 0

if flag==1:
    
    ### load data for plotting
    input_flg = '01' # 10, 01, 11
    file = '../results/data/moments/data_perturbations_' + input_flg + '.pickle'
    
    with open(file,'rb') as f:
        [_, _, _, dev_prediction_steady, dev_variance_steady] = pickle.load(f)
    
    ### plot deviations
    plot_deviations_upon_perturbations(dev_prediction_steady, dev_variance_steady)
    

# %% Plot perturbation summary 

flag = 0
import matplotlib.pyplot as plt
import seaborn as sns

if flag==1:
    
    ### load data for plotting
    file_10 = '../results/data/moments/data_perturbations_10.pickle'
    file_01 = '../results/data/moments/data_perturbations_01.pickle'
    file_11 = '../results/data/moments/data_perturbations_11.pickle'
    
    with open(file_10,'rb') as f:
        [_, _, _, dev_prediction_steady_10, dev_variance_steady_10] = pickle.load(f)
        
    with open(file_01,'rb') as f:
        [_, _, _, dev_prediction_steady_01, dev_variance_steady_01] = pickle.load(f)
        
    with open(file_11,'rb') as f:
        [_, _, _, dev_prediction_steady_11, dev_variance_steady_11] = pickle.load(f)
        
    
    ### if deviation is smaller than 2% 
    # dev_prediction_steady_10[abs(dev_prediction_steady_10)<2] = 0
    # dev_prediction_steady_01[abs(dev_prediction_steady_01)<2] = 0
    # dev_prediction_steady_11[abs(dev_prediction_steady_11)<2] = 0
    
    # dev_variance_steady_10[abs(dev_variance_steady_10)<2] = 0
    # dev_variance_steady_01[abs(dev_variance_steady_01)<2] = 0
    # dev_variance_steady_11[abs(dev_variance_steady_11)<2] = 0
        
    ### restructure data
    data_prediction_exc_perturbation = np.zeros((8,3))
    data_prediction_exc_perturbation[:,0] = np.sign(dev_prediction_steady_10[1,:])
    data_prediction_exc_perturbation[:,1] = np.sign(dev_prediction_steady_01[1,:])
    data_prediction_exc_perturbation[:,2] = np.sign(dev_prediction_steady_11[1,:])
    
    data_prediction_inh_perturbation = np.zeros((8,3))
    data_prediction_inh_perturbation[:,0] = np.sign(dev_prediction_steady_10[0,:])
    data_prediction_inh_perturbation[:,1] = np.sign(dev_prediction_steady_01[0,:])
    data_prediction_inh_perturbation[:,2] = np.sign(dev_prediction_steady_11[0,:])
    
    data_variance_exc_perturbation = np.zeros((8,3))
    data_variance_exc_perturbation[:,0] = np.sign(dev_variance_steady_10[1,:])
    data_variance_exc_perturbation[:,1] = np.sign(dev_variance_steady_01[1,:])
    data_variance_exc_perturbation[:,2] = np.sign(dev_variance_steady_11[1,:])
    
    data_variance_inh_perturbation = np.zeros((8,3))
    data_variance_inh_perturbation[:,0] = np.sign(dev_variance_steady_10[0,:])
    data_variance_inh_perturbation[:,1] = np.sign(dev_variance_steady_01[0,:])
    data_variance_inh_perturbation[:,2] = np.sign(dev_variance_steady_11[0,:])
    
    ### plot data
    plot_heatmap_perturbation_all(data_prediction_exc_perturbation, xticklabels=False, yticklabels=True)
    plot_heatmap_perturbation_all(data_prediction_inh_perturbation, xticklabels=True, yticklabels=False)
    plot_heatmap_perturbation_all(data_variance_exc_perturbation, xticklabels=False, yticklabels=False)
    plot_heatmap_perturbation_all(data_variance_inh_perturbation, xticklabels=True, yticklabels=False)
    