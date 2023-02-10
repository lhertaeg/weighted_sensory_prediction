#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:31:58 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.default_parameters import default_para_mfn
from src.functions_networks import run_mfn_circuit, run_mfn_circuit_coupled

dtype = np.float32

# %% functions

def stimuli_moments_from_uniform(n_trials, num_values_per_trial, min_mean, max_mean, m_sd, n_sd):
    
    mean_stimuli = np.random.uniform(min_mean, max_mean, size=n_trials)
    sd_stimuli = np.maximum(mean_stimuli * m_sd + n_sd, 0) ######### in csae n_sd is negative !!!
    
    stimuli = np.array([], dtype=dtype)
    
    for id_stim in range(n_trials):
        
        inputs_per_stimulus = np.random.normal(mean_stimuli[id_stim], sd_stimuli[id_stim], size=num_values_per_trial)
        stimuli = np.concatenate((stimuli, inputs_per_stimulus))
        
    return dtype(stimuli)



def random_binary_from_moments(mean, sd, n_stimuli, pa=0.5):
    
    if ((sd!=0) & (mean!=0)):
        a = mean - sd * np.sqrt((1-pa)/pa)
        b = mean + sd * np.sqrt(pa/(1-pa))
        rnd = dtype(np.random.choice([a,b],size=n_stimuli,p=[pa,1-pa]))
    else:
        rnd = np.zeros(n_stimuli, dtype=dtype)
        
    return rnd


def random_gamma_from_moments(mean, sd, n_stimuli):
    
    if ((sd!=0) & (mean!=0)):
        shape = mean**2 / sd**2
        scale = sd**2  / mean 
        rnd = dtype(np.random.gamma(shape, scale, size=n_stimuli))
    else:
        rnd = np.zeros(n_stimuli, dtype=dtype)
        
    return rnd


def random_lognormal_from_moments(mean, sd, n_stimuli):
    
    if ((sd!=0) & (mean!=0)):
        a = np.log(mean**2/np.sqrt(mean**2 + sd**2))
        b = np.sqrt(np.log(sd**2 / mean**2 + 1))
        rnd = dtype(np.random.lognormal(a, b, size=n_stimuli))
    else:
        rnd = np.zeros(n_stimuli, dtype=dtype)
        
    return rnd


def random_uniform_from_moments(mean, sd, num):
    
    b = np.sqrt(12) * sd / 2 + mean
    a = 2 * mean -b
    rnd = dtype(np.random.uniform(a, b, size = num))
        
    return rnd


def simulate_example_pe_circuit(mfn_flag, mean_stimuli, std_stimuli, file_for_data, 
                                seed = 186, trial_duration = np.int32(100000), 
                                num_values_per_trial = np.int32(200), dist_type = 'uniform', pa=0.8):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
    
    ### create stimuli
    np.random.seed(seed)
    repeats_per_value = trial_duration//num_values_per_trial

    if dist_type=='uniform':
        stimuli = random_uniform_from_moments(mean_stimuli, std_stimuli, num_values_per_trial)
    elif dist_type=='normal':
        stimuli = np.random.normal(mean_stimuli, std_stimuli, size=num_values_per_trial)
    elif dist_type=='lognormal':
        stimuli = random_lognormal_from_moments(mean_stimuli, std_stimuli, num_values_per_trial)
    elif dist_type=='gamma':
        stimuli = random_gamma_from_moments(mean_stimuli, std_stimuli, num_values_per_trial)
    elif dist_type=='binary_equal_prop':
        stimuli = random_binary_from_moments(mean_stimuli, std_stimuli, num_values_per_trial)
    elif dist_type=='binary_unequal_prop':
        stimuli = random_binary_from_moments(mean_stimuli, std_stimuli, num_values_per_trial, pa=pa)
        
    stimuli = np.repeat(stimuli, repeats_per_value)
    
    ### compute variances and predictions
    
    ## run model
    prediction, variance, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                              fixed_input, stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)

    ### save data for later
    with open(file_for_data,'wb') as f:
        pickle.dump([mean_stimuli, std_stimuli, trial_duration, num_values_per_trial, stimuli, prediction, variance],f)    
        
    return [mean_stimuli, std_stimuli, trial_duration, num_values_per_trial, stimuli, prediction, variance]



def simulate_pe_uniform_para_sweep(mfn_flag, means_tested, variances_tested, file_for_data, 
                                    seed = 186, trial_duration = np.int32(100000), 
                                    num_values_per_trial = np.int32(200)):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
      v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
      tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
    
    ### initialise
    mse_mean = np.zeros((len(means_tested), len(variances_tested), trial_duration), dtype=dtype)
    mse_variance = np.zeros((len(means_tested), len(variances_tested), trial_duration), dtype=dtype)
    
    ### parameter sweep
    for i, mean_dist in enumerate(means_tested):
        
            print('Mean:', mean_dist)
            
            for j, var_dist in enumerate(variances_tested):
    
                print('-- Variance:', var_dist)
                
                ## create stimuli
                np.random.seed(seed)
                repeats_per_value = trial_duration//num_values_per_trial
                stimuli = random_uniform_from_moments(mean_dist, np.sqrt(var_dist), num_values_per_trial)
                stimuli = np.repeat(stimuli, repeats_per_value)
                
                ## run model
                prediction, variance, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                          fixed_input, stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
                
                ## compute mean squared error
                running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1, dtype=dtype)
                mse_mean[i, j, :] = (running_average - prediction)**2
                
                mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1, dtype=dtype)
                momentary_variance = (stimuli - mean_running)**2
                running_variance = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1, dtype=dtype)
                mse_variance[i, j, :] = (running_variance - variance)**2
                
    ### save data for later
    with open(file_for_data,'wb') as f:
        pickle.dump([trial_duration, num_values_per_trial, means_tested, 
                      variances_tested, mse_mean, mse_variance],f)
        
    return [trial_duration, num_values_per_trial, means_tested, variances_tested, mse_mean, mse_variance]
         

def simulate_weighting_example(mfn_flag, min_mean, max_mean, m_sd, n_sd, seed = np.int32(186), n_trials = np.int32(100), 
                               trial_duration = np.int32(5000), num_values_per_trial = np.int32(10), file_for_data = None):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ### create stimuli
    np.random.seed(seed)
    n_repeats_per_stim = dtype(trial_duration/num_values_per_trial)
    
    stimuli = stimuli_moments_from_uniform(n_trials, num_values_per_trial, min_mean, max_mean, m_sd, n_sd)
    stimuli = np.repeat(stimuli, n_repeats_per_stim)
    
    ### run model
    [m_neuron_lower, v_neuron_lower, m_neuron_higher, v_neuron_higher, 
     alpha, beta, weighted_output] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                             v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                             tc_var_pred, tau_pe, fixed_input, stimuli, 
                                                             VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                             v_PE_to_V = v_PE_to_V)                                                              

    ### save data for later
    if file_for_data is not None:
        
        with open(file_for_data,'wb') as f:
            pickle.dump([n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, 
                         v_neuron_lower, m_neuron_higher, v_neuron_higher, alpha, beta, weighted_output],f)
            
    return [n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, 
            v_neuron_lower, m_neuron_higher, v_neuron_higher, alpha, beta, weighted_output]



def simulate_weighting_exploration(mfn_flag, variability_within, variability_across, mean_trials, m_sd, last_n = np.int32(30),
                                   seed = np.int32(186), n_trials = np.int32(100), trial_duration = np.int32(5000), 
                                   num_values_per_trial = np.int32(10), file_for_data = None):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ### initialise
    weight = np.zeros((len(variability_within),len(variability_across)), dtype=dtype)

    ### exploration (run model for different input statistics)
    for col, std_mean in enumerate(variability_across):
        
        print('Variability across trials:', std_mean)
        
        for row, n_sd in enumerate(variability_within):
            
            ## display progress
            print('-- Variability within trial:', n_sd)
    
            ## define stimuli
            np.random.seed(seed)
            n_repeats_per_stim = dtype(trial_duration/num_values_per_trial)
    
            stimuli = stimuli_moments_from_uniform(n_trials, num_values_per_trial, dtype(mean_trials - np.sqrt(3)*std_mean), 
                                                   dtype(mean_trials + np.sqrt(3)*std_mean), dtype(m_sd), dtype(n_sd))
            stimuli = np.repeat(stimuli, n_repeats_per_stim)

            ## run model
            [m_neuron_lower, v_neuron_lower, m_neuron_higher, v_neuron_higher, 
             alpha, beta, weighted_output] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                                     v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                                     tc_var_pred, tau_pe, fixed_input, stimuli, 
                                                                     VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                                     v_PE_to_V = v_PE_to_V)   
                                                                     
            ### fraction of sensory input in weighted output
            weight[row, col] = np.mean(alpha[-last_n * trial_duration:])

    
    ### save data for later
    with open(file_for_data,'wb') as f:
        pickle.dump([variability_within, variability_across, weight],f) 
        
    return [variability_within, variability_across, weight]
            
     
        
def simulate_dynamic_weighting_eg(mfn_flag, min_mean_before, max_mean_before, m_sd_before, n_sd_before, 
                                  min_mean_after, max_mean_after, m_sd_after, n_sd_after, seed = np.int32(186),
                                  n_trials = np.int32(120), trial_duration = np.int32(5000), num_values_per_trial = np.int32(10),
                                  file_for_data = None):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ### create stimuli
    np.random.seed(seed)
    n_repeats_per_stim = dtype(trial_duration/num_values_per_trial)
    
    stimuli = np.zeros(n_trials * num_values_per_trial, dtype=dtype)
    mid = np.int32((n_trials * num_values_per_trial)//2)
    
    stimuli[:mid] = stimuli_moments_from_uniform(n_trials//2, num_values_per_trial, min_mean_before, max_mean_before, 
                                                m_sd_before, n_sd_before)
    stimuli[mid:] = stimuli_moments_from_uniform(n_trials//2, num_values_per_trial, min_mean_after, max_mean_after, 
                                                m_sd_after, n_sd_after)
    
    stimuli = np.repeat(stimuli, n_repeats_per_stim)
    
    
    ### run model
    [m_neuron_lower, v_neuron_lower, m_neuron_higher, v_neuron_higher, 
     alpha, beta, weighted_output] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                             v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                             tc_var_pred, tau_pe, fixed_input, stimuli, 
                                                             VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                             v_PE_to_V = v_PE_to_V)                                                              

    ### save data for later
    if file_for_data is not None:
        
        with open(file_for_data,'wb') as f:
            pickle.dump([n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, 
                         v_neuron_lower, m_neuron_higher, v_neuron_higher, alpha, beta, weighted_output],f)
            
    return [n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, 
            v_neuron_lower, m_neuron_higher, v_neuron_higher, alpha, beta, weighted_output]


def simulate_sensory_weight_time_course(mfn_flag, variability_within, variability_across, mean_trials, 
                                        m_sd, seed = np.int32(186), trial_duration = np.int32(5000),
                                        n_trials = np.int32(100), num_values_per_trial = np.int32(10),
                                        file_for_data = None):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ### initialise
    weight = np.zeros((n_trials * trial_duration, len(variability_across)), dtype=dtype)

    ### exploration (run model for different input statistics)
    for id_stim in range(len(variability_across)):
        
        std_mean = variability_across[id_stim]
        std_std = variability_within[id_stim]
        
        ## display progress
        print('Variability across trials:', std_mean)
        print('-- Variability within trial:', std_std)
    
        ## define stimuli
        np.random.seed(seed)
        n_repeats_per_stim = trial_duration/num_values_per_trial

        stimuli = stimuli_moments_from_uniform(n_trials, num_values_per_trial, dtype(mean_trials - np.sqrt(3)*std_mean), 
                                               dtype(mean_trials + np.sqrt(3)*std_mean), dtype(m_sd), dtype(std_std))
        stimuli = np.repeat(stimuli, n_repeats_per_stim)

        ## run model
        [m_neuron_lower, v_neuron_lower, m_neuron_higher, v_neuron_higher, 
         alpha, beta, weighted_output] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                                 v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                                 tc_var_pred, tau_pe, fixed_input, stimuli, 
                                                                 VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                                 v_PE_to_V = v_PE_to_V)   
                                                                 
        ### fraction of sensory input in weighted output
        weight[:, id_stim] = alpha


    ### save data for later
    if file_for_data is not None:
        
        with open(file_for_data,'wb') as f:
            pickle.dump([n_trials, variability_within, variability_across, weight],f) 
            
    return [n_trials, variability_within, variability_across, weight]


def simulate_impact_para(mfn_flag, variability_within, variability_across, mean_trials, 
                         m_sd, last_n = np.int32(30), seed = np.int32(186), 
                         n_trials = np.int32(100), trial_duration = np.int32(5000), 
                         num_values_per_trial = np.int32(10), file_for_data = None,
                         n = 2, gain_w_PE_to_P = 1, gain_v_PE_to_P = 1, add_input = 0,
                         id_cells_modulated = np.array([True,True,False,False,True,True,True,True])):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    w_PE_to_P *= gain_w_PE_to_P
    v_PE_to_P *= gain_v_PE_to_P
    
    if add_input!=0:
        perturbation = np.zeros((n_trials * trial_duration, 8))                          
        perturbation[(n_trials * trial_duration)//2:, id_cells_modulated] = add_input  
        fixed_input_plus = fixed_input + perturbation
    else:
        fixed_input_plus = fixed_input
    
    ### initialise
    weight = np.zeros(len(variability_across), dtype=dtype)

    ### exploration (run model for different input statistics)
    for id_stim in range(len(variability_across)):
        
        std_mean = variability_across[id_stim]
        std_std = variability_within[id_stim]
        
        ## display progress
        print('Variability across trials:', std_mean)
        print('-- Variability within trial:', std_std)
        
        ## define stimuli
        np.random.seed(seed)
        n_repeats_per_stim = trial_duration/num_values_per_trial

        stimuli = stimuli_moments_from_uniform(n_trials, num_values_per_trial, dtype(mean_trials - np.sqrt(3)*std_mean), 
                                               dtype(mean_trials + np.sqrt(3)*std_mean), dtype(m_sd), dtype(std_std))
        stimuli = np.repeat(stimuli, n_repeats_per_stim)

        ## run model
        [m_neuron_lower, v_neuron_lower, m_neuron_higher, v_neuron_higher, 
         alpha, beta, weighted_output] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                                 v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                                 tc_var_pred, tau_pe, fixed_input_plus, stimuli, 
                                                                 VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                                 v_PE_to_V = v_PE_to_V, n=n)
        
        
        ### comppute steady state
        weight[id_stim] = np.mean(alpha[-last_n * trial_duration:])
        
    ### save data for later
    if file_for_data is not None:
        
        with open(file_for_data,'wb') as f:
            pickle.dump([stimuli, n, gain_w_PE_to_P, gain_v_PE_to_P, 
                         add_input, id_cells_modulated, weight],f) 
            
    return [stimuli, n, gain_w_PE_to_P, gain_v_PE_to_P, add_input, id_cells_modulated, weight]



def simulate_neuromod(mfn_flag, std_mean, n_sd, column, xp, xs, xv, mean_trials = dtype(5), m_sd = dtype(0), 
                      last_n = np.int32(50), seed = np.int32(186), n_trials = np.int32(200), trial_duration = np.int32(5000), 
                      num_values_per_trial = np.int32(10), file_for_data = None, mult_input = dtype(1)):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
      v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
      tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ### define stimuli
    np.random.seed(seed)
    n_repeats_per_stim = dtype(trial_duration/num_values_per_trial)

    stimuli = stimuli_moments_from_uniform(n_trials, num_values_per_trial, dtype(mean_trials - np.sqrt(3)*std_mean), 
                                            dtype(mean_trials + np.sqrt(3)*std_mean), dtype(m_sd), dtype(n_sd))
    stimuli = np.repeat(stimuli, n_repeats_per_stim)
    
    ### run model without perturbation
    print('Without neuromodulation\n')
    [_, _, _, _, alpha, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                        v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                        tc_var_pred, tau_pe, fixed_input, stimuli, 
                                                        VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                        v_PE_to_V = v_PE_to_V)
    
    alpha_before_pert = np.mean(alpha[(n_trials - last_n) * trial_duration:])
    
    ### initialise
    nums = np.size(xp,0)
    alpha_after_pert = np.zeros((nums, nums), dtype=dtype)

    ### run model for different fractions of INs activated
    print('With neuromodulation:\n')
    
    for i in range(nums):
        for j in range(nums):
        
            if ~np.isnan(xv[i,j]):
                ## display progress
                print('-- xp:', xp[i,j], 'xs:', xs[i,j], 'xv:', xv[i,j], 'r:', xp[i,j]**2 + xs[i,j]**2 + xv[i,j]**2)
                
                ## add perturbation XXXX
                perturbation = np.zeros((n_trials * trial_duration,8), dtype=dtype)                          
                perturbation[(n_trials * trial_duration)//2:, 4:6] = xp[i,j] * mult_input
                perturbation[(n_trials * trial_duration)//2:, 6] = xs[i,j] * mult_input
                perturbation[(n_trials * trial_duration)//2:, 7] = xv[i,j] * mult_input
                fixed_input_plus_perturbation = fixed_input + perturbation
                    
                if column==1:
                    fixed_input_lower = fixed_input_plus_perturbation
                    fixed_input_higher = fixed_input
                elif column==2:
                    fixed_input_lower = fixed_input
                    fixed_input_higher = fixed_input_plus_perturbation
                elif column==0:
                    fixed_input_lower = fixed_input_plus_perturbation
                    fixed_input_higher = fixed_input_plus_perturbation
        
                ## run model
                [m_neuron_lower, v_neuron_lower, m_neuron_higher, v_neuron_higher, 
                  alpha, beta, weighted_output] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                                          v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                                          tc_var_pred, tau_pe, None, stimuli, VS = VS,
                                                                          VV = VV, w_PE_to_V = w_PE_to_V, v_PE_to_V = v_PE_to_V, 
                                                                          fixed_input_lower = fixed_input_lower,
                                                                          fixed_input_higher = fixed_input_higher)
                
                
                ### comppute steady state
                alpha_after_pert[i, j] = np.mean(alpha[-last_n * trial_duration:])
                
            else:
                
                alpha_after_pert[i, j] = np.nan
        
    ### save data for later
    if file_for_data is not None:
        
        with open(file_for_data,'wb') as f:
            pickle.dump([xp, xs, xv, alpha_before_pert, alpha_after_pert],f) 
            
    return [xp, xs, xv, alpha_before_pert, alpha_after_pert]