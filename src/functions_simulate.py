#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:31:58 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle
import matplotlib.pyplot as plt

from src.default_parameters import default_para_mfn
from src.functions_networks import run_mfn_circuit, run_mfn_circuit_coupled, run_spatial_mfn_circuit

dtype = np.float32

# %% functions

def stimuli_moments_from_uniform(n_trials, num_values_per_trial, min_mean, max_mean, m_sd, n_sd, out_trial_means=False, natural_numbers=False):
    
    mean_stimuli = np.random.uniform(min_mean, max_mean, size=n_trials)
    sd_stimuli = np.maximum(mean_stimuli * m_sd + n_sd, 0) ######### in case n_sd is negative !!!
    
    if natural_numbers:
        mean_stimuli = np.round(mean_stimuli)
    
    stimuli = np.array([], dtype=dtype)
    
    for id_stim in range(n_trials):
        
        inputs_per_stimulus = np.random.normal(mean_stimuli[id_stim], sd_stimuli[id_stim], size=num_values_per_trial)
        stimuli = np.concatenate((stimuli, inputs_per_stimulus))
     
    results = (dtype(stimuli),)
    if out_trial_means:
        results += (mean_stimuli,)
        
    return results


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
    a = 2 * mean - b
    rnd = dtype(np.random.uniform(a, b, size = num))
        
    return rnd



def simulate_PE_circuit_P_fixed_S_constant(mfn_flag, initial_prediction, constant_stimuli, file_for_data, 
                                           trial_duration = np.int32(100000), num_values_per_trial = np.int32(200)):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, _, _, _, _, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
    
    ### take out w_PE_to_P
    w_PE_to_P *= 0
    
    ### initialise
    nPE = np.zeros(len(constant_stimuli))
    pPE = np.zeros(len(constant_stimuli))
    
    ### test all stimuli
    for i, stimulus in enumerate(constant_stimuli):
        
        print('Stimulus ', i+1, '/', len(constant_stimuli))
    
        repeats_per_value = trial_duration//num_values_per_trial
        stimuli = random_uniform_from_moments(stimulus, 0, num_values_per_trial)
        stimuli = np.repeat(stimuli, repeats_per_value)
    
        ## run model
        prediction, variance, rate_pe = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                        fixed_input, stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V,
                                                        pred_ini = initial_prediction)

        ## save 
        nPE[i] = rate_pe[-1,0]
        pPE[i] = rate_pe[-1,1]
  
    return nPE, pPE


def simulate_spatial_example(mfn_flag, mean_stimulus, spatial_std, file_for_data, num_sub_nets = np.int32(100), seed = 186, 
                             num_time_steps = np.int32(1000), dist_type = 'uniform', pa=0.8, M_init = None, V_init = None, rates_init = None):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
    
    ### create spatial noise
    np.random.seed(seed)

    if dist_type=='uniform':
        spatial_noise = random_uniform_from_moments(0, spatial_std, num_sub_nets)
        scaling = 2/(num_sub_nets * np.sqrt(3))
    elif dist_type=='normal':
        spatial_noise = np.random.normal(0, spatial_std, size=num_sub_nets)
        scaling = 1/num_sub_nets # needs to be computed
        #scaling = 2/(num_sub_nets * np.sqrt(3))
    elif dist_type=='lognormal':
        spatial_noise = random_lognormal_from_moments(0, spatial_std, num_sub_nets)
        scaling = 1/num_sub_nets # needs to be computed
    elif dist_type=='gamma':
        spatial_noise = random_gamma_from_moments(0, spatial_std, num_sub_nets)
        scaling = 1/num_sub_nets # needs to be computed
    elif dist_type=='binary_equal_prop':
        spatial_noise = random_binary_from_moments(0, spatial_std, num_sub_nets)
        scaling = 1/num_sub_nets # needs to be computed
    elif dist_type=='binary_unequal_prop':
        spatial_noise = random_binary_from_moments(0, spatial_std, num_sub_nets, pa=pa)
        scaling = 1/num_sub_nets # needs to be computed
        
    spatial_noise = np.repeat(spatial_noise, 8)

    ### connectivity matrices
    W_PE_to_V = np.tile(np.array([1.015,1.023,0,0,0,0,0,0]), (1,num_sub_nets)) * scaling
    W_PE_to_P = 100 * np.tile(w_PE_to_P, (1,num_sub_nets)) / num_sub_nets
    W_P_to_PE = np.tile(w_P_to_PE, (num_sub_nets,1))
    W_PE_to_PE = np.kron(np.eye(num_sub_nets,dtype=dtype),w_PE_to_PE)
    
    tc_var_per_stim /= 10 # !!!!!!!!!!!!!!!!!!!!!!!!!
    
    ### run model
    if rates_init is None:
        m_neuron, v_neuron, rates_final = run_spatial_mfn_circuit(W_PE_to_V, W_PE_to_P, W_P_to_PE, W_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                                  fixed_input, mean_stimulus, spatial_noise, VS=VS, VV=VV, 
                                                                  num_time_steps = num_time_steps, num_sub_nets = num_sub_nets)
    else:
        m_neuron, v_neuron, rates_final = run_spatial_mfn_circuit(W_PE_to_V, W_PE_to_P, W_P_to_PE, W_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                                  fixed_input, mean_stimulus, spatial_noise, VS=VS, VV=VV, 
                                                                  num_time_steps = num_time_steps, num_sub_nets = num_sub_nets,
                                                                  M_init = M_init, V_init = V_init, rates_init = rates_init)

    ### save data for later
    with open(file_for_data,'wb') as f:
        pickle.dump([mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron, v_neuron, rates_final],f)    
        
    return [mean_stimulus, spatial_std, spatial_noise, num_time_steps, m_neuron, v_neuron, rates_final]



def simulate_slope_vs_variances(mfn_flag, min_mean, max_mean_arr, m_sd, n_sd_arr, trial_duration = np.int32(5000), seed = np.int32(186), n_trials = np.int32(100),
                                num_values_per_trial = np.int32(10), num_trial_ss = np.int32(30), natural_numbers=False, file_for_data = None):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ### 
    if max_mean_arr.ndim == 0:
        max_mean = max_mean_arr
        values_tested = n_sd_arr
    else:
        n_sd = n_sd_arr
        values_tested = max_mean_arr
    
    ### initialise
    fitted_slopes = np.zeros(len(values_tested), dtype=dtype)
    
    ### run over all trial durations
    for i, value in enumerate(values_tested):
        
        ### display progress & set seed
        print(value)
        np.random.seed(seed)
        
        ### create stimuli
        n_repeats_per_stim = dtype(trial_duration/num_values_per_trial)
        
        if max_mean_arr.ndim == 0:
            stimuli, trial_means = stimuli_moments_from_uniform(n_trials, num_values_per_trial, min_mean, max_mean, m_sd, value, out_trial_means = True, 
                                                                natural_numbers = natural_numbers)   
        else:
            stimuli, trial_means = stimuli_moments_from_uniform(n_trials, num_values_per_trial, min_mean, value, m_sd, n_sd, out_trial_means = True, 
                                                                natural_numbers = natural_numbers) 
            
        stimuli = np.repeat(stimuli, n_repeats_per_stim)
        
        ### run model
        [m_neuron_lower, v_neuron_lower, m_neuron_higher, v_neuron_higher, 
         alpha, beta, weighted_output] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                                 v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                                 tc_var_pred, tau_pe, fixed_input, stimuli, 
                                                                 VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                                 v_PE_to_V = v_PE_to_V) 
                                                                 
        ### extract slope
        trials_sensory = np.mean(np.split(stimuli, n_trials),1)[num_trial_ss:]
        trials_estimated = np.mean(np.split(weighted_output, n_trials),1)[num_trial_ss:]
        p = np.polyfit(trials_sensory, trials_estimated, 1)
        fitted_slopes[i] = p[0]

    ### save data for later
    if file_for_data is not None:
        
        with open(file_for_data,'wb') as f:
            pickle.dump([values_tested, fitted_slopes],f)
    
    ### get the results
    return [values_tested, fitted_slopes]



def simulate_slope_vs_trial_duration(mfn_flag, min_mean, max_mean, m_sd, n_sd, trial_durations, seed = np.int32(186), n_trials = np.int32(100),
                  num_values_per_trial = np.int32(10), num_trial_ss = np.int32(30), natural_numbers=False, file_for_data = None):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ### initialise
    fitted_slopes = np.zeros(len(trial_durations), dtype=dtype)
    
    ### run over all trial durations
    for i, trial_duration in enumerate(trial_durations):
        
        ### display progress & set seed
        print(trial_duration)
        np.random.seed(seed)
        
        ### create stimuli
        n_repeats_per_stim = dtype(trial_duration/num_values_per_trial)
        
        stimuli, trial_means = stimuli_moments_from_uniform(n_trials, num_values_per_trial, min_mean, max_mean, m_sd, n_sd, out_trial_means = True, 
                                                            natural_numbers = natural_numbers)   
        stimuli = np.repeat(stimuli, n_repeats_per_stim)
        
        ### run model
        [m_neuron_lower, v_neuron_lower, m_neuron_higher, v_neuron_higher, 
         alpha, beta, weighted_output] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                                 v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                                 tc_var_pred, tau_pe, fixed_input, stimuli, 
                                                                 VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                                 v_PE_to_V = v_PE_to_V) 
                                                                 
        ### extract slope
        trials_sensory = np.mean(np.split(stimuli, n_trials),1)[num_trial_ss:]
        trials_estimated = np.mean(np.split(weighted_output, n_trials),1)[num_trial_ss:]
        p = np.polyfit(trials_sensory, trials_estimated, 1)
        fitted_slopes[i] = p[0]

    ### save data for later
    if file_for_data is not None:
        
        with open(file_for_data,'wb') as f:
            pickle.dump([trial_durations, fitted_slopes],f)
    
    ### get the results
    return [trial_durations, fitted_slopes]



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
                                    num_values_per_trial = np.int32(200), record_interneuron_activity = False,
                                    record_pe_activity = False):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
      v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
      tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
    
    ### initialise
    dev_mean = np.zeros((len(means_tested), len(variances_tested)), dtype=dtype)
    dev_variance = np.zeros((len(means_tested), len(variances_tested)), dtype=dtype)
    activity_interneurons = np.zeros((len(means_tested), len(variances_tested), trial_duration, 4), dtype=dtype)
    activity_pe_neurons = np.zeros((len(means_tested), len(variances_tested), trial_duration, 2), dtype=dtype)
    
    ### parameter to compute the steady state quantities
    n_ss = 3 * trial_duration//4
    
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
                if record_interneuron_activity:
                    prediction, variance, rates_pe , rates_ints = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                                                  fixed_input, stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V, 
                                                                                  record_interneuron_activity = record_interneuron_activity)
                    
                    # extract the activity of all the interneurons in the system
                    activity_interneurons[i, j, :, :] = rates_ints
                
                else:
                    prediction, variance, rates_pe,  = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                                       fixed_input, stimuli, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
                
                # extract the activity of nPE and pPE
                if record_pe_activity:
                    activity_pe_neurons[i, j, :, :] = rates_pe
                
                ## compute mean squared error between running average/variance and m or v neuron
                running_average = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1, dtype=dtype)
                dev_mean[i, j] = (np.mean(running_average[n_ss:]) - np.mean(prediction[n_ss:])) / np.mean(running_average[n_ss:])
                
                mean_running = np.cumsum(stimuli)/np.arange(1,len(stimuli)+1, dtype=dtype)
                momentary_variance = (stimuli - mean_running)**2
                running_variance = np.cumsum(momentary_variance)/np.arange(1,len(stimuli)+1, dtype=dtype)
                dev_variance[i, j] = (np.mean(running_variance[n_ss:]) - np.mean(variance[n_ss:])) / np.mean(running_variance[n_ss:])
                
                
    ### save data for later
    if (record_pe_activity & record_interneuron_activity):
        
        with open(file_for_data,'wb') as f:
            pickle.dump([trial_duration, num_values_per_trial, means_tested, variances_tested, dev_mean, dev_variance, 
                         activity_pe_neurons, activity_interneurons],f)
        
    elif (record_pe_activity & ~record_interneuron_activity): 
        
        with open(file_for_data,'wb') as f:
            pickle.dump([trial_duration, num_values_per_trial, means_tested, variances_tested, dev_mean, dev_variance, 
                         activity_pe_neurons],f)
            
    elif (~record_pe_activity & record_interneuron_activity): 
        
        with open(file_for_data,'wb') as f:
            pickle.dump([trial_duration, num_values_per_trial, means_tested, variances_tested, dev_mean, dev_variance, 
                         activity_interneurons],f)          
    else:
        
        with open(file_for_data,'wb') as f:
            pickle.dump([trial_duration, num_values_per_trial, means_tested, variances_tested, dev_mean, dev_variance],f)
      
    ### return results
    ret = (trial_duration, num_values_per_trial, means_tested, variances_tested, dev_mean, dev_variance)
    
    if record_pe_activity:
        ret += (activity_pe_neurons, )
    
    if record_interneuron_activity:
        ret += (activity_interneurons, )     
      
    return ret
         

def simulate_weighting_example(mfn_flag, min_mean, max_mean, m_sd, n_sd, seed = np.int32(186), n_trials = np.int32(100), 
                               trial_duration = np.int32(5000), num_values_per_trial = np.int32(10), natural_numbers=False,
                               file_for_data = None):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ### create stimuli
    np.random.seed(seed)
    n_repeats_per_stim = dtype(trial_duration/num_values_per_trial)
    
    stimuli, trial_means = stimuli_moments_from_uniform(n_trials, num_values_per_trial, min_mean, max_mean, m_sd, n_sd, out_trial_means = True, 
                                                        natural_numbers = natural_numbers)   
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
                         v_neuron_lower, m_neuron_higher, v_neuron_higher, alpha, beta, weighted_output, trial_means],f)
    
    ### get the results
    results = (n_trials, trial_duration, num_values_per_trial, stimuli, m_neuron_lower, 
               v_neuron_lower, m_neuron_higher, v_neuron_higher, alpha, beta, weighted_output, trial_means)
    
    return results


def simulate_weighting_exploration(mfn_flag, variability_within, variability_across, mean_trials, m_sd, last_n = np.int32(30),
                                   seed = np.int32(186), n_trials = np.int32(100), trial_duration = np.int32(5000), 
                                   num_values_per_trial = np.int32(10), file_for_data = None, record_interneuron_activity=False):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ### initialise
    weight = np.zeros((len(variability_within),len(variability_across)), dtype=dtype)
    activity_interneurons_lower = np.zeros((len(variability_within),len(variability_across), 4), dtype=dtype)
    activity_interneurons_higher = np.zeros((len(variability_within),len(variability_across), 4), dtype=dtype)

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
            if record_interneuron_activity:
                [m_neuron_lower, v_neuron_lower, m_neuron_higher, 
                 v_neuron_higher, alpha, beta, weighted_output, 
                 rates_int_lower, rates_int_higher] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                                              v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                                              tc_var_pred, tau_pe, fixed_input, stimuli, 
                                                                              VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                                              v_PE_to_V = v_PE_to_V, record_interneuron_activity=record_interneuron_activity)   
              
                activity_interneurons_lower[row, col, :] = np.mean(rates_int_lower[-last_n * trial_duration:,:],0)
                activity_interneurons_higher[row, col, :] = np.mean(rates_int_higher[-last_n * trial_duration:,:],0)
                                                                         
            else:
                [m_neuron_lower, v_neuron_lower, m_neuron_higher, v_neuron_higher, 
                 alpha, beta, weighted_output] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                                         v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                                         tc_var_pred, tau_pe, fixed_input, stimuli, 
                                                                         VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                                         v_PE_to_V = v_PE_to_V)  
                                                                         
                                                                     
            ### fraction of sensory input in weighted output
            weight[row, col] = np.mean(alpha[-last_n * trial_duration:])

    
    ### save data for later
    if record_interneuron_activity: 
        with open(file_for_data,'wb') as f:
            pickle.dump([variability_within, variability_across, weight, 
                         activity_interneurons_lower, activity_interneurons_higher],f) 
    else:
        with open(file_for_data,'wb') as f:
            pickle.dump([variability_within, variability_across, weight],f) 
        
        
    ### return results
    ret = (variability_within, variability_across, weight,)
    
    if record_interneuron_activity:
        ret += (activity_interneurons_lower, activity_interneurons_higher, )     
      
    return ret    


def simulate_activity_neurons(mfn_flag, variability_within, variability_across, mean_trials, m_sd, seeds,
                                    last_n = np.int32(30), n_trials = np.int32(100), trial_duration = np.int32(5000), 
                                    num_values_per_trial = np.int32(10), file_for_data = None):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ### initialise
    activity_interneurons_lower = np.zeros((len(variability_within),len(variability_across), len(seeds), 4), dtype=dtype)
    activity_interneurons_higher = np.zeros((len(variability_within),len(variability_across), len(seeds), 4), dtype=dtype)
    activity_pe_neurons_lower = np.zeros((len(variability_within),len(variability_across), len(seeds), 2), dtype=dtype)
    activity_pe_neurons_higher = np.zeros((len(variability_within),len(variability_across), len(seeds), 2), dtype=dtype)

    ### exploration (run model for different input statistics)
    for k, seed in enumerate(seeds):
        
        print('Seed:', seed)
    
        for col, std_mean in enumerate(variability_across):
            
            print('- Variability across trials:', std_mean)
            
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
                [m_neuron_lower, v_neuron_lower, m_neuron_higher, 
                 v_neuron_higher, alpha, beta, weighted_output,
                 rates_pe_lower, rates_pe_higher,
                 rates_int_lower, rates_int_higher] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                                              v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                                              tc_var_pred, tau_pe, fixed_input, stimuli, 
                                                                              VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                                              v_PE_to_V = v_PE_to_V, record_pe_activity = True,
                                                                              record_interneuron_activity=True)   
              
                activity_interneurons_lower[row, col, k, :] = np.mean(rates_int_lower[-last_n * trial_duration:,:],0)
                activity_interneurons_higher[row, col, k,  :] = np.mean(rates_int_higher[-last_n * trial_duration:,:],0)
                activity_pe_neurons_lower[row, col, k, :] = np.mean(rates_pe_lower[-last_n * trial_duration:,:],0)
                activity_pe_neurons_higher[row, col, k,  :] = np.mean(rates_pe_higher[-last_n * trial_duration:,:],0)
                                                                             
    
    ### save data for later
    with open(file_for_data,'wb') as f:
        pickle.dump([variability_within, variability_across, activity_pe_neurons_lower, activity_pe_neurons_higher,
                     activity_interneurons_lower, activity_interneurons_higher],f) 
        
    ### return results       
    return [variability_within, variability_across, activity_pe_neurons_lower, activity_pe_neurons_higher, 
            activity_interneurons_lower, activity_interneurons_higher]       
      
     
        
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


def simulate_neuromod_combos(mfn_flag, std_mean, n_sd, column, xp, xs, xv, mean_trials = dtype(5), m_sd = dtype(0), 
                             last_n = np.int32(50), seed = np.int32(186), n_trials = np.int32(200), 
                             trial_duration = np.int32(5000), num_values_per_trial = np.int32(10), 
                             file_for_data = None, mult_input = dtype(1)):
    
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
    nums = len(xp)
    alpha_after_pert = np.zeros(nums, dtype=dtype)

    ### run model for different fractions of INs activated
    print('With neuromodulation:\n')
    
    for i in range(nums):
        
        ## display progress
        print('-- xp:', xp[i], 'xs:', xs[i], 'xv:', xv[i])
        
        ## add perturbation XXXX
        perturbation = np.zeros((n_trials * trial_duration,8), dtype=dtype)                          
        perturbation[(n_trials * trial_duration)//2:, 4:6] = xp[i] * mult_input
        perturbation[(n_trials * trial_duration)//2:, 6] = xs[i] * mult_input
        perturbation[(n_trials * trial_duration)//2:, 7] = xv[i] * mult_input
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
        alpha_after_pert[i] = np.mean(alpha[-last_n * trial_duration:])
                
        
    ### save data for later
    if file_for_data is not None:
        
        with open(file_for_data,'wb') as f:
            pickle.dump([xp, xs, xv, alpha_before_pert, alpha_after_pert],f) 
            
    return [xp, xs, xv, alpha_before_pert, alpha_after_pert]



def simulate_moment_estimation_upon_changes_PE(mfn_flag, std_mean, n_sd, column, pert_stength, mean_trials = dtype(5), m_sd = dtype(0), 
                                               last_n = np.int32(50), seed = np.int32(186), n_trials = np.int32(200), trial_duration = np.int32(5000), 
                                               num_values_per_trial = np.int32(10), file_for_data = None):
    
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
    
    ### initialise
    nums = len(pert_stength)
    m_act_lower = np.zeros((len(pert_stength), 2), dtype=dtype)
    v_act_lower = np.zeros((len(pert_stength), 2), dtype=dtype)
    v_act_higher = np.zeros((len(pert_stength), 2), dtype=dtype)

    ### run model for input onto PE neuron
    for cell_id in range(2):
        
        print('PE neuron type ', cell_id)
        
        for i in range(nums):
                
            ## display progress
            print('-- Perturbation strength:', pert_stength[i])
            
            ## add perturbation XXXX
            perturbation = np.zeros((n_trials * trial_duration,8), dtype=dtype)                          
            perturbation[(n_trials * trial_duration)//2:, cell_id] = pert_stength[i]
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
        
            
            ### compute steady state
            m_act_lower[i, cell_id] = np.mean(m_neuron_lower[-last_n * trial_duration:])
            v_act_lower[i, cell_id] = np.mean(v_neuron_lower[-last_n * trial_duration:])
            v_act_higher[i, cell_id] = np.mean(v_neuron_higher[-last_n * trial_duration:])
            
        
    ### save data for later
    if file_for_data is not None:
        
        with open(file_for_data,'wb') as f:
            pickle.dump([pert_stength, m_act_lower, v_act_lower, v_act_higher],f) 
            
    return [pert_stength, m_act_lower, v_act_lower, v_act_higher]


def simulate_neuromod_effect_on_neuron_properties(mfn_flag, min_mean, max_mean, m_sd, n_sd, id_cell = None, pert_stength = dtype(1),
                                                  seed = np.int32(186), n_trials = np.int32(100), trial_duration = np.int32(5000),
                                                  num_values_per_trial = np.int32(10), file_for_data = None, plot_data=False):
    
    ### load default parameters
    VS, VV = int(mfn_flag[0]), int(mfn_flag[1])
    
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ### activate neuron with pert_stength if id_cell is not none
    if id_cell is not None:
        perturbation = np.zeros((n_trials * trial_duration,8), dtype=dtype)                          
        perturbation[:, id_cell] = pert_stength
        fixed_input_plus_perturbation = fixed_input + perturbation
            
        fixed_input_lower = fixed_input_plus_perturbation # we only look at the lower PE circuit later, so we apply perturbation there
        fixed_input_higher = fixed_input
        
    else:
        fixed_input_lower = fixed_input
        fixed_input_higher = fixed_input
      
    
    ### create stimuli
    np.random.seed(seed)
    n_repeats_per_stim = dtype(trial_duration/num_values_per_trial)
    
    stimuli = stimuli_moments_from_uniform(n_trials, num_values_per_trial, min_mean, max_mean, m_sd, n_sd)
    stimuli = np.repeat(stimuli, n_repeats_per_stim)
    
    ### run model
    [m_neuron_lower, v_neuron_lower, m_neuron_higher, 
     v_neuron_higher, alpha, beta, weighted_output, 
     PE_lower, PE_higher] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                    v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                    tc_var_pred, tau_pe, None, stimuli, 
                                                    VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                    v_PE_to_V = v_PE_to_V, 
                                                    fixed_input_lower = fixed_input_lower,
                                                    fixed_input_higher = fixed_input_higher,
                                                    record_pe_activity = True) 
    
    ### extract BL and gain of nPE and pPE neurons                                          
    n_begin = trial_duration // num_values_per_trial - 1
    n_every = trial_duration // num_values_per_trial
    
    p_minus_s = m_neuron_lower[n_begin::n_every] - stimuli[n_begin::n_every]
    s_minus_p = (stimuli[n_begin::n_every] - m_neuron_lower[n_begin::n_every])
    nPE = PE_lower[n_begin::n_every,0]
    pPE = PE_lower[n_begin::n_every,1]
    
    gain_nPE, baseline_nPE = np.polyfit(p_minus_s[(p_minus_s>=0) & (p_minus_s<2.5)], 
                                        nPE[(p_minus_s>=0) & (p_minus_s<2.5)], 1) 
    
    gain_pPE, baseline_pPE = np.polyfit(s_minus_p[(s_minus_p>=0) & (s_minus_p<2.5)], 
                                        pPE[(s_minus_p>=0) & (s_minus_p<2.5)], 1)
    
    if plot_data:
        plt.figure()
        plt.plot(p_minus_s, nPE, '.')
        
        plt.figure()
        plt.plot(s_minus_p, pPE, '.')
        
    
    return [baseline_nPE, baseline_pPE, gain_nPE, gain_pPE]

