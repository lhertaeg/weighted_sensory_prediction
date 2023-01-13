#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import



# %% Example: PE neurons can estimate/establish the mean and variance of a normal random variable 
# mean and varaince are drawn from a unimodal distribution

flag = 0
flg_plot_only = 1

if flag==1:
    
    ### define/load parameters
    input_flg = '10' # 10, 01, 11
    
    VS, VV = int(input_flg[0]), int(input_flg[1])
    filename = '../results/data/moments/Data_Optimal_Parameters_MFN_' + input_flg + '.pickle'
    file_data4plot = '../results/data/moments/Data_example_MFN_' + input_flg + '.pickle'
    
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
        n_trials = 500
        trial_duration = 500
        n_stimuli_per_trial = 1
        n_repeats_per_stim = trial_duration/n_stimuli_per_trial
        
        mean_stimuli, std_stimuli = 5, 2
        stimuli = random_uniform_from_moments(mean_stimuli, std_stimuli, (n_trials * n_stimuli_per_trial))
        stimuli = np.repeat(stimuli, n_repeats_per_stim)
        
        ### compute variances and predictions
        
        ## run model
        prediction, variance, PE_activity = run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                            tc_var_per_stim, tau_pe, fixed_input, 
                                                                            stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
    
        ### save data for later
        with open(file_data4plot,'wb') as f:
            pickle.dump([n_trials, trial_duration, stimuli, prediction, variance],f)                                      
     
    else:
        ### load data for plotting
        with open(file_data4plot,'rb') as f:
            [n_trials, trial_duration, stimuli, prediction, variance] = pickle.load(f)  
        
        
    ### plot prediction and variance (in comparison to the data) 
    plot_prediction(n_trials, stimuli, trial_duration, prediction)  
    plot_variance(n_trials, stimuli, trial_duration, variance) 