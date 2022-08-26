#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 09:48:42 2022

@author: loreen.hertaeg
"""

# %% Import

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from src.mean_field_model import stimuli_moments_from_uniform


# %% limit cases for Kalman filter (F=H=1)

flag = 0
flg_limit_case = 1 # 0 = mean the same, std large; 1 = mean varies, std = 0

if flag==1:
    
    ### define stimuli
    n_trials = 100
    trial_duration = 5000
    n_stimuli_per_trial = 10
    n_repeats_per_stim = trial_duration/n_stimuli_per_trial
    # In each trial a stimulus is shown. This stimulus may vary (depending on limit case)
    # Between trials the stimulus is either the same or varies (depending on the limit case)
        
    if flg_limit_case==0:
        stimuli = stimuli_moments_from_uniform(n_trials, np.int32(n_stimuli_per_trial), 3, 3, 1, 5) # mean 3, SD between 1 and 5
    else:
        stimuli = stimuli_moments_from_uniform(n_trials, np.int32(n_stimuli_per_trial), 1, 5, 0, 0) # mean between 1 and 5, SD 0
  
    stimuli = np.repeat(stimuli, n_repeats_per_stim)
    
    ### define Kalman parameters
    sigmaNoise = 1#1e-100
    sigmaPrior = 1
    
    ### apply Kalman filter
    estimated_mean = 0
    estimated_var = sigmaPrior**2
    
    estimated_means = []
    estimated_vars = []
    
    for i in range(len(stimuli)):
    
        K = estimated_var / (estimated_var + sigmaNoise**2)
        estimated_mean += K * (stimuli[i] - estimated_mean) 
        estimated_var *= (1 - K)
        
        estimated_vars.append(estimated_var)
        estimated_means.append(estimated_mean)
    
    
    ### plot results
    plt.figure()
    plt.plot(stimuli)
    plt.plot(estimated_means)
    

# limit case '0' works ofc
# limit case '1' does not work
# reason:
    # prediction = K * y + (1-K) * prediction
    # if I want it to follow y completely, K must be one
    # that means sigmaNoise would need to be almost zero
    # however, as soon as K=1, estimated_var becomes zero
    # hence, in the next iteration step K becomes zero
    # and consequently, estimated mean would not change anymore
    
    
# %% limit cases for Kalman filter (including F and H)

flag = 1
flg_limit_case = 0 # 0 = mean the same, std large; 1 = mean varies, std = 0

if flag==1:
    
    ### define stimuli
    n_trials = 100
    trial_duration = 5000
    n_stimuli_per_trial = 10
    n_repeats_per_stim = trial_duration/n_stimuli_per_trial
    # In each trial a stimulus is shown. This stimulus may vary (depending on limit case)
    # Between trials the stimulus is either the same or varies (depending on the limit case)
        
    if flg_limit_case==0:
        stimuli = stimuli_moments_from_uniform(n_trials, np.int32(n_stimuli_per_trial), 3, 3, 1, 5) # mean 3, SD between 1 and 5
    else:
        stimuli = stimuli_moments_from_uniform(n_trials, np.int32(n_stimuli_per_trial), 1, 5, 0, 0) # mean between 1 and 5, SD 0
  
    stimuli = np.repeat(stimuli, n_repeats_per_stim)
    
    ### define Kalman parameters
    sigmaNoise = 1
    sigmaPrior = 1
    
    F = 1
    H = 1
    
    ### apply Kalman filter
    estimated_mean = 0
    estimated_var = sigmaPrior**2
    
    estimated_means = []
    estimated_vars = []
    
    for i in range(len(stimuli)):
        
        # Propagate (actually "prediction")
        if i > 0:
            estimated_mean *= F
            estimated_var *= F**2
    
        # Estimate (actually "correction")
        K = estimated_var * H / (H * estimated_var * H + sigmaNoise**2)
        estimated_mean += K * (stimuli[i] - H * estimated_mean)
        estimated_var *= (1 - K * H)
        
        estimated_vars.append(estimated_var)
        estimated_means.append(estimated_mean)
    
    
    ### plot results
    plt.figure()
    plt.plot(stimuli)
    plt.plot(estimated_means)
    
    
# H != 1:
    # limit case '1' can still not be accounted for
    # to make sure that out ~ s, K must be one
    # however, if K=1, (1-K*H) would only be one if H=1 => contradiction
    # if we want to make sure that (1-K*H) = 0, then making sigmaNoise almost zero does the trick
    # however, then K=1/H => out would be a fraction of s
    # moreover, estimated_var would be zero again => same problem as above
    
# F != 1:
    # F<0 ... prediction would vanish, if sigmaNoise almost zero
    # F>0 ... prediction would explode, if sigmaNoise almost zero
    
# Both F and H not 1:
    # limit case '1': F = 1.1 and H = 1.9 for noise parameters = 1 gave reasonable results
    # limit case '0': F = 1 and H = 1 for noise parameters = 1 gave reasonable results
    # that already tells us that F and H, as well as sigma's must be chosen wisely
    # and that transitions would require adapting F and H appropriately!
    