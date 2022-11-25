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
from src.mean_field_model import run_mean_field_model_one_column, run_pe_circuit_mfn

from src.toy_model import random_uniform_from_moments, random_lognormal_from_moments, random_gamma_from_moments
from src.plot_toy_model import plot_limit_case, plot_alpha_para_exploration_ratios, plot_fraction_sensory_comparsion, plot_alpha_para_exploration
from src.plot_toy_model import plot_manipulation_results

from src.plot_results_mfn import plot_prediction, plot_variance, plot_mse, plot_manipulations, plot_deviations_upon_perturbations
from src.plot_results_mfn import plot_heatmap_perturbation_all, plot_deviation_vs_effect_size, plot_deviation_vs_PE
from src.plot_results_mfn import plot_deviation_vs_PE_II

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Perturbing the E/I balance in PE circuits - example

flag = 0
flg_plot_only = 0

if flag==1:
    
    ### load and define parameters
    input_flg = '10' # 10, 01, 11
    
    VS, VV = int(input_flg[0]), int(input_flg[1])
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/perturbations/data_perturbation_example_' + input_flg + '.pickle'
    
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
        # best undestood as one trial in which 1000 stimuli are shown, each 500 (ms) long (stimuli drawn from distribution)
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
    input_flg = '10' # 10, 01, 11
    
    VS, VV = int(input_flg[0]), int(input_flg[1])
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/perturbations/data_perturbations_' + input_flg + '.pickle'
    
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
    file = '../results/data/perturbations/data_perturbations_' + input_flg + '.pickle'
    
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
    file_10 = '../results/data/perturbations/data_perturbations_10.pickle'
    file_01 = '../results/data/perturbations/data_perturbations_01.pickle'
    file_11 = '../results/data/perturbations/data_perturbations_11.pickle'
    
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
    

# %% Estimation bias and BL activity

# Please note that when BL activity of nPE (pPE) is increased the BL of the other PE neuron increases equivalently
# (dendrites are balanced in FB phase => nPE/pPE receive roughly the same amount of additional excitation)
# in MM and PB phase though, because dendrites may be excited or inhibited, this is not necessarily true
# => effect size of nPE to pPE changes which leads to estimation bias

flag = 0
flg_plot_only = 1

if flag==1:
    
    file_data4plot = '../results/data/perturbations/data_deviations_vs_BL.pickle'
    
    if flg_plot_only==0:
        
        ### load and define parameters
        input_flg = '10' # 10, 01, 11
        
        VS, VV = int(input_flg[0]), int(input_flg[1])
        filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
        
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
        # best undestood as one trial in which 1000 stimuli are shown, each 500 (ms) long (stimuli drawn from distribution)
        n_trials = 1000     
        trial_duration = 500
        min_trial_for_avg = 750
        n_stimuli_per_trial = 1
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        
        mean_stimuli, std_stimuli = 5, 2
        stimuli = random_uniform_from_moments(mean_stimuli, std_stimuli, (n_trials * n_stimuli_per_trial))
        stimuli = np.repeat(stimuli, n_repeats_per_stim)
        fixed_input = np.tile(fixed_input, (len(stimuli),1))
        
        injected_inputs = np.arange(5)
        
        ### initialise
        dev_prediction_steady = np.zeros((2, len(injected_inputs)))
        dev_variance_steady = np.zeros((2, len(injected_inputs)))
        nPE_BL = np.zeros((2, len(injected_inputs)))
        pPE_BL = np.zeros((2, len(injected_inputs)))
        
        ### add perturbation
        for id_cell in range(2): # BL of nPE (0) or pPE (1)
        
            for i, add_input_strength in enumerate(injected_inputs):
                
                ### add input to one of the PE neurons to emulate an increased BL activity
                add_input = np.zeros((len(stimuli),8))                          
                add_input[(n_trials * trial_duration)//2:, id_cell] = add_input_strength          
                fixed_input_plus = fixed_input + add_input
                
                ### compute BL activity
                _, _, PE_activity = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                    tc_var_per_stim, tau_pe, fixed_input_plus, 
                                                                    np.zeros_like(stimuli), VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
                
                
                nPE_BL[id_cell, i] = PE_activity[-1, 0]
                pPE_BL[id_cell, i] = PE_activity[-1, 1]
                
        
                ### compute variances and predictions
                prediction, variance, _ = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                                    tc_var_per_stim, tau_pe, fixed_input_plus, 
                                                                                    stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
    
    
                running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
                dev_prediction = (prediction - running_average) / running_average
                
                mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1)
                momentary_variance = (stimuli - mean_running)**2
                variance_running = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1)
                dev_variance = (variance - variance_running) / variance_running
                
                dev_prediction_steady[id_cell, i] = np.mean(dev_prediction[(min_trial_for_avg * trial_duration):]) * 100
                dev_variance_steady[id_cell, i] = np.mean(dev_variance[(min_trial_for_avg * trial_duration):]) * 100
        
        
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([nPE_BL, pPE_BL, dev_prediction_steady, dev_variance_steady],f)                                    
     
    else:
        
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [nPE_BL, pPE_BL, dev_prediction_steady, dev_variance_steady] = pickle.load(f)
        
            
    ### plot
    plt.figure()
    plt.plot(nPE_BL[0, :] - pPE_BL[0, :], dev_prediction_steady[0, :])
    
    plt.figure()
    plt.plot(nPE_BL[1, :] - pPE_BL[1, :], dev_prediction_steady[1, :])
    
    plt.figure()
    plt.plot(nPE_BL[0, :] - pPE_BL[0, :], dev_variance_steady[0, :])
    
    plt.figure()
    plt.plot(nPE_BL[1, :] - pPE_BL[1, :], dev_variance_steady[1, :])
    

# %% Estimation bias depends on effect size of nPE:pPE neurons 

flag = 0
flg_plot_only = 1

if flag==1:
    
    file_data4plot = '../results/data/perturbations/data_deviations_vs_effect_size.pickle'
    
    if flg_plot_only==0:
        
        ### load and define parameters
        input_flg = '10' # 10, 01, 11
        
        VS, VV = int(input_flg[0]), int(input_flg[1])
        filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
        
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
        n_trials = 500
        trial_duration = 500
        min_trial_for_avg = 300
        n_stimuli_per_trial = 1
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        
        mean_stimuli, std_stimuli = 5, 2
        stimuli = random_uniform_from_moments(mean_stimuli, std_stimuli, (n_trials * n_stimuli_per_trial))
        stimuli = np.repeat(stimuli, n_repeats_per_stim)
        
        ### compute variances and predictions for equal effect size
        prediction_0, variance_0, _ = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                  tc_var_per_stim, tau_pe, fixed_input, 
                                                                  stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
        
        ### scaling fcators to be tested
        scaling_factors = np.linspace(0.5,2,7)
        
        ### initialise
        dev_prediction_steady = np.zeros((2, len(scaling_factors)))
        dev_variance_steady = np.zeros((2, len(scaling_factors)))
        
        ### compute variances and predictions after effect size has been changed
        for id_cell in range(2): # effect size of nPE (0) or pPE (1)
        
            for i, scaling_factor in enumerate(scaling_factors):
                
                [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                 tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename, VS=VS, VV=VV)
                
                ## change effect size
                if input_flg=='10':
                    nPE_scale = 1.015
                    pPE_scale = 1.023
                elif input_flg=='01':
                    nPE_scale = 1.7 # 1.72
                    pPE_scale = 1.7 # 1.68
                elif input_flg=='11':
                    nPE_scale = 2.49
                    pPE_scale = 2.53
                    
                if id_cell==0:
                    nPE_scale *= scaling_factor
                elif id_cell==1:
                    pPE_scale *= scaling_factor
                    
                w_PE_to_P[0,0] *= nPE_scale
                w_PE_to_P[0,1] *= pPE_scale
                w_PE_to_V = [nPE_scale, pPE_scale]
                
                
                ## run model
                prediction, variance, _ = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                          tc_var_per_stim, tau_pe, fixed_input_plus, 
                                                                          stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
    
                ### compute deviations
                dev_prediction = (prediction - prediction_0) / prediction_0
                dev_variance = (variance - variance_0) / variance_0
                
                dev_prediction_steady[id_cell, i] = np.mean(dev_prediction[(min_trial_for_avg * trial_duration):]) * 100
                dev_variance_steady[id_cell, i] = np.mean(dev_variance[(min_trial_for_avg * trial_duration):]) * 100
        
        
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([scaling_factors, dev_prediction_steady, dev_variance_steady],f)                                    
     
    else:
        
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [scaling_factors, dev_prediction_steady, dev_variance_steady] = pickle.load(f)
        
            
    # ### plot
    plot_deviation_vs_effect_size(scaling_factors, dev_prediction_steady, 'Bias in mean', plot_legend=False)
    plot_deviation_vs_effect_size(scaling_factors, dev_variance_steady, 'Bias in variance')

    
# %% Estimate how strongly/much a neuron is driven by (S-P) or (P-S) for all 3 MFN

flag = 0

if flag==1:
    
    input_flgs = ['10', '01', '11']
    
    for input_flg in input_flgs:
    
        ### load and define parameters
        VS, VV = int(input_flg[0]), int(input_flg[1])
        filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
        
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename, VS=VS, VV=VV)
        
        ### define stimulus length
        num_steps = 5000
        feedforward_strengths = np.linspace(1,5,5)
        feedback_strengths = np.linspace(1,5,5)
        
        rates_steady_state_feedforward = np.zeros((len(feedforward_strengths), 8))
        rates_steady_state_feedback = np.zeros((len(feedback_strengths), 8))
    
        ### test different sensory input strengths
        prediction = np.zeros(num_steps)
    
        for i, feedforward_strength in enumerate(feedforward_strengths):
            
            stimuli = feedforward_strength * np.ones(num_steps)
    
            ###  run network
            rates = run_pe_circuit_mfn(w_PE_to_PE, tau_pe, fixed_input, stimuli, 
                                       prediction, VS = VS, VV = VV, dt = dtype(1))
            
            ### steady state activity
            rates_steady_state_feedforward[i, :] = rates[-1,:]
            
        
        ### extract slope of linear fit for each cell type
        slopes_feedforward = []
        
        for i in [0,1,4,5,6,7]:
            if sum(rates_steady_state_feedforward[:,i]>0)>0:
                m, n = np.polyfit(feedforward_strengths[rates_steady_state_feedforward[:,i]>0], 
                                  rates_steady_state_feedforward[rates_steady_state_feedforward[:,i]>0,i], 1)
            else:
                m = 0
            slopes_feedforward.append(m)
            
            #plt.figure()
            #plt.plot(feedforward_strengths, rates_steady_state_feedforward[:,i], 'r')
            
        
        ### test different prediction strengths
        stimuli = np.zeros(num_steps)
    
        for i, feedback_strength in enumerate(feedback_strengths):
            
            prediction = feedback_strength * np.ones(num_steps)
    
            ###  run network
            rates = run_pe_circuit_mfn(w_PE_to_PE, tau_pe, fixed_input, stimuli, 
                                        prediction, VS = VS, VV = VV, dt = dtype(1))
            
            ### steady state activity
            rates_steady_state_feedback[i, :] = rates[-1,:]
            
        
        ### extract slope of linear fit for each cell type
        slopes_feedback = []
        
        for i in [0,1,4,5,6,7]:
            if sum(rates_steady_state_feedback[:,i]>0)>0:
                m, n = np.polyfit(feedback_strengths[rates_steady_state_feedback[:,i]>0], 
                                  rates_steady_state_feedback[rates_steady_state_feedback[:,i]>0,i], 1)
            else:
                m = 0
            slopes_feedback.append(m)
            
            #plt.figure()
            #plt.plot(feedforward_strengths, rates_steady_state_feedback[:,i], 'b')
  
        ## save data for later
        file_save = '../results/data/perturbations/data_neuron_drive_' + input_flg + '.pickle'
        
        with open(file_save,'wb') as f:
            pickle.dump([feedforward_strengths, feedback_strengths, slopes_feedforward, slopes_feedback],f)   


# %% Estimate deviation direction by how strongly/much a neuron is driven by sensory input or prediction

flag = 0

if flag==1:
    
    input_flgs = ['10', '01', '11']
    marker = ['o', 's', 'D']
    labels = ['MFN 1', 'MFN 2', 'MFN 3']
    moment_flg = 1 # 0 = mean, 1 = variance
    
    
    plot_deviation_vs_PE(moment_flg, input_flgs, marker, labels, perturbation_direction=-1, 
                         plot_deviation_gradual = False)
    
    plot_deviation_vs_PE(moment_flg, input_flgs, marker, labels, perturbation_direction=1, 
                         plot_deviation_gradual = True)
    
    plot_deviation_vs_PE(moment_flg, input_flgs, marker, labels, perturbation_direction=-1, 
                         plot_deviation_gradual = True)
    
 
    
# %% Estimate net impact of each neuron on both nPE and pPE neurons

# Please note that the linear fit is only an approximation. Due to non-linearities (e.g. rectifications)
# the data would sometimes be better fit by two lines (one for inhibitory stimulation, one for 
# excitatory stimulation). Hence, it is only an approximation. Most of the time, it does not matter.
# However, for MFN 01 (SOM and VIP) could be a bit off (for the variance).
# Either fix it by showing results for inh/exc stimulatoin separately or simply explain in text.

flag = 0

if flag==1:
    
    input_flgs = ['10', '01', '11']
    
    for input_flg in input_flgs:
    
        ### load and define parameters
        VS, VV = int(input_flg[0]), int(input_flg[1])
        filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
        file_data4plot = '../results/data/perturbations/data_deviations_pe_vs_stimulation.pickle'
        
        [w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para(filename, VS=VS, VV=VV)
        
        ### define stimulus length and strength
        num_steps = 5000
        stimulation_stengths = np.linspace(-0.5,0.5,9)

        ### initialise
        fixed_input = np.tile(fixed_input, (num_steps,1))
        rates_steady_state_nPE = np.zeros((len(stimulation_stengths), 6))
        rates_steady_state_pPE = np.zeros((len(stimulation_stengths), 6))
        
        ### nPE activity
        prediction = 5 * np.ones(num_steps)
        stimuli = np.zeros(num_steps) 
    
        for i, stimulation_strength in enumerate(stimulation_stengths):
            
            for j, cell_id in enumerate([0,1,4,5,6,7]): # 6 targets
            
                stim_input = np.zeros((len(stimuli),8))                          
                stim_input[:, cell_id] = stimulation_strength          
                fixed_input_plus = fixed_input + stim_input
    
                ###  run network
                rates = run_pe_circuit_mfn(w_PE_to_PE, tau_pe, fixed_input_plus, stimuli, 
                                           prediction, VS = VS, VV = VV, dt = dtype(1))
            
                ### steady state activity
                rates_steady_state_nPE[i, j] = rates[-1,0]
               
                
        ### pPE activity
        prediction = np.zeros(num_steps) 
        stimuli = 5 * np.ones(num_steps) 
    
        for i, stimulation_strength in enumerate(stimulation_stengths):
            
            for j, cell_id in enumerate([0,1,4,5,6,7]): # 6 targets
            
                stim_input = np.zeros((len(stimuli),8))                          
                stim_input[:, cell_id] = stimulation_strength          
                fixed_input_plus = fixed_input + stim_input
    
                ###  run network
                rates = run_pe_circuit_mfn(w_PE_to_PE, tau_pe, fixed_input_plus, stimuli, 
                                           prediction, VS = VS, VV = VV, dt = dtype(1))
            
                ### steady state activity
                rates_steady_state_pPE[i, j] = rates[-1,1]
        
        ### extract slope of linear fit for each cell type
        slopes_nPE = []
        slopes_pPE = []
        
        for j, cell_id in enumerate([0,1,4,5,6,7]):
            # nPE neurons
            m, n = np.polyfit(stimulation_stengths, rates_steady_state_nPE[:,j], 1)
            slopes_nPE.append(m)
            
            #plt.plot(stimulation_stengths, rates_steady_state_nPE[:,j])
            
            # nPE neurons
            m, n = np.polyfit(stimulation_stengths, rates_steady_state_pPE[:,j], 1)
            slopes_pPE.append(m)
            
            plt.plot(stimulation_stengths, rates_steady_state_pPE[:,j])
            
            
        ## save data for later
        file_save = '../results/data/perturbations/data_pe_vs_neuron_stim_' + input_flg + '.pickle'
        
        with open(file_save,'wb') as f:
            pickle.dump([stimulation_stengths, slopes_nPE, slopes_pPE],f) 
            
 
# %% Estimate deviation direction by how strongly/much a neuron is driven by sensory input or prediction

flag = 1

if flag==1:
    
    input_flgs = ['10', '01', '11']
    marker = ['o', 's', 'D']
    labels = ['MFN 1', 'MFN 2', 'MFN 3']
    moment_flg = 1 # 0 = mean, 1 = variance
    
    
    plot_deviation_vs_PE_II(moment_flg, input_flgs, marker, labels, perturbation_direction=1, 
                            plot_deviation_gradual = False)
    
    # plot_deviation_vs_PE_II(moment_flg, input_flgs, marker, labels, perturbation_direction=1, 
    #                         plot_deviation_gradual = True)
    
    # plot_deviation_vs_PE_II(moment_flg, input_flgs, marker, labels, perturbation_direction=-1, 
    #                         plot_deviation_gradual = True)
    
 
