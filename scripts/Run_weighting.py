#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:32:45 2022

@author: loreen.hertaeg
"""


# %% Import

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from src.Default_values import Default_PredProcPara
from src.Functions_Network import Neurons, Network, Activity_Zero, InputStructure, Stimulation, Simulation, RunStaticNetwork
from src.Functions_Network import Stimulation_new, RunStaticNetwork_new, sensory_input_euler
from src.Functions_PredNet import RunPredNet, Neurons, Network, InputStructure, Stimulation, Simulation, Activity_Zero

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Toy model - rethink model

flag = 1
flg = 1

if flag==1:
    
    ### stimuli
    n_stimuli = 20
    if flg==0:
        mean_stimuli = np.random.uniform(5, 5, size=n_stimuli)
        sd_stimuli = np.random.uniform(1, 5, size=n_stimuli)
    else:
        mean_stimuli = np.random.uniform(1, 5, size=n_stimuli)
        sd_stimuli = np.random.uniform(0, 0, size=n_stimuli)
    stimulus_duration = 1000
    stimuli = np.array([])
    
    for i in range(n_stimuli):
        
        inputs_per_stimulus = np.random.normal(mean_stimuli[i], sd_stimuli[i], size=stimulus_duration)
        stimuli = np.concatenate((stimuli, inputs_per_stimulus))
    
    
    ### compute variances and predictions
    
    # parameters
    tc_mean_per_stim = 20
    tc_var_per_stim = 20
    eta_prediction = 5e-3
    eta_mean_prediction = 2e-4
    tc_var_pred = 1
    
    # initialise
    pred = 0
    var_pred = 0
    mean_per_stim = 0
    var_per_stim = 0
    mean_prediction = 0
    
    prediction = np.zeros_like(stimuli)
    mean_per_stimulus = np.zeros_like(stimuli)
    variance_per_stimulus = np.zeros_like(stimuli)
    mean_of_prediction = np.zeros_like(stimuli)
    variance_prediction = np.zeros_like(stimuli)
    
    for id_stim, stim in enumerate(stimuli):
    
        # # compute mean per trial - smoothed version of input
        # mean_per_stim = (1-1/tc_mean_per_stim) * mean_per_stim + stim/tc_mean_per_stim
        # mean_per_stimulus[id_stim] = mean_per_stim
        
        # # compute variance per trial -- compute nPE and pPE activity and feed into excitatory neuron that smoothes their activity
        # pPE = (np.maximum(stim - mean_per_stim,0))**2
        # nPE = (np.maximum(mean_per_stim - stim,0))**2
        
        # var_per_stim = (1-1/tc_var_per_stim) * var_per_stim + (nPE+pPE)/tc_var_per_stim
        # variance_per_stimulus[id_stim] = var_per_stim
        
        # compute prediction ("running mean over stimuli")
        pPE_sensory = (np.maximum(stim - pred,0))**2   
        nPE_sensory = (np.maximum(pred - stim,0))**2
        pred += eta_prediction * (pPE_sensory - nPE_sensory)
        prediction[id_stim] = pred
        
        # compute variance of sensory input
        var_per_stim = (1-1/tc_var_per_stim) * var_per_stim + (nPE_sensory+pPE_sensory)/tc_var_per_stim
        variance_per_stimulus[id_stim] = var_per_stim
        
        # compute variance of prediction
        pPE_prediction = (np.maximum(pred - mean_prediction,0))**2
        nPE_prediction = (np.maximum(mean_prediction - pred,0))**2
        mean_prediction += eta_mean_prediction * (pPE_prediction - nPE_prediction)
        mean_of_prediction[id_stim] = mean_prediction
        
        var_pred = (1-1/tc_var_pred) * var_pred + (nPE_prediction + pPE_prediction)/tc_var_pred
        variance_prediction[id_stim] = var_pred
       
       
    ### compute weighted output
    alpha = (1/variance_per_stimulus) / ((1/variance_per_stimulus) + (1/variance_prediction))
    beta = (1/variance_prediction) / ((1/variance_per_stimulus) + (1/variance_prediction))
    weighted_output = alpha * stimuli + beta * prediction
       
    ### plot results
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12,10))
    
    for i in range(n_stimuli):
        ax1.axvline((i+1)*stimulus_duration, ls='--', color='k', alpha=0.2)
    ax1.plot(stimuli, color='#D76A03', label='stimulus')
    #ax1.plot(mean_per_stimulus, color='g')
    ax1.plot(prediction, color='#19535F', label='prediction')
    ax1.plot(mean_of_prediction, color='#BFCC94', label='mean of prediction')
    ax1.axhline(np.mean(stimuli), ls=':')
    ax1.set_xlim([0,len(stimuli)])
    ax1.set_ylabel('Activity')
    ax1.set_title('Sensory inputs and predictions')
    ax1.legend(loc=0, ncol=3)
    sns.despine(ax=ax1)
    
    
    var_per_stimulus = np.var(np.array_split(stimuli, n_stimuli),1)
    for i in range(n_stimuli):
        ax2.axvline((i+1)*stimulus_duration, ls='--', color='k', alpha=0.2)
    for i in range(n_stimuli):
        ax2.axhline(var_per_stimulus[i], i/n_stimuli, (i+1)/n_stimuli, color='g')
    ax2.plot(variance_per_stimulus, color='#D76A03', label='var(stimulus)')
    ax2.plot(variance_prediction, color='#19535F', label='var(prediction)')
    #ax2.axhline(np.var(stimuli), ls=':')
    ax2.set_xlim([0,len(stimuli)])
    ax2.set_ylabel('Activity')
    ax2.set_title('Variances of sensory inputs and predictions')
    ax2.legend(loc=0, ncol=2)
    sns.despine(ax=ax2)
    
    for i in range(n_stimuli):
        ax3.axvline((i+1)*stimulus_duration, ls='--', color='k', alpha=0.2)
    ax3.plot(stimuli, color='#D76A03', label='stimulus')
    ax3.plot(prediction, color='#19535F', label='prediction')
    ax3.plot(weighted_output, color='#582630', label='weighted output')
    ax3.set_xlim([0,len(stimuli)])
    ax3.set_ylabel('Activity')
    ax3.set_xlabel('Time (ms)')
    ax3.set_title('Weighted output compared to sensory inputs & predictions')
    ax3.legend(loc=0, ncol=3)
    sns.despine(ax=ax3)
    

# %% Toy model - weighting

flag = 0

if flag==1:
    
    N = 500 # number of trials
    S1 = np.random.uniform(0, 10, size=N)
    S2 = np.random.uniform(3, 7, size=N)
    S3 = np.random.uniform(2, 8, size=N)
    S4 = np.random.uniform(4, 6, size=N)
    S5 = np.random.uniform(0, 10, size=N)
    S = np.concatenate((S1, S2, S3, S4, S5))
    
    eta = 1e-4
    eta_SD = 1e-5
    tau = 100
    tau_SD = 10
    E = 0
    SD = 0
    SD_long = 0
    E_SD = 0
    
    rate_1 = np.zeros_like(S)
    rate_2 = np.zeros_like(S)
    rate_3 = np.zeros_like(S)
    
    for i, s in enumerate(S):
        
        pPE = (np.maximum(s - E,0))**2
        nPE = (np.maximum(E - s,0))**2
        E_SD = (1-1/tau) * E_SD + s/tau
        SD = (1-1/tau_SD) * SD + (nPE+pPE)/tau_SD
        
        for t in np.arange(500):
            
            pPE_mean = (np.maximum(s - E,0))**2
            nPE_mean = (np.maximum(E - s,0))**2
            E += eta * (pPE_mean - nPE_mean)/tau
            
            # pPE = (np.maximum(s - E,0))**2
            # nPE = (np.maximum(E - s,0))**2
            # E_SD = (1-1/tau) * E_SD + s/tau
            # SD = (1-1/tau_SD) * SD + (nPE+pPE)/tau_SD
            
            pPE_SD = (np.maximum(SD - SD_long,0))**2
            nPE_SD = (np.maximum(SD_long - SD,0))**2
            SD_long += eta_SD * (pPE_SD - nPE_SD)/tau
                  
        rate_1[i] = E
        rate_2[i] = SD
        rate_3[i] = SD_long
    
    ### compute weighted output
    var_s = rate_2
    var_p = rate_3
    alpha = (1/var_s) / ((1/var_s) + (1/var_p))
    beta = (1/var_p) / ((1/var_s) + (1/var_p))
    
    out = alpha * S + beta * rate_1
    
    ### plot
    fig = plt.figure(figsize=(10,8))#, tight_layout=True)
    fig.subplots_adjust(hspace=0.5)
    
    ax = plt.subplot(3,1,1)
    ax.plot(S, 'r')
    ax.plot(rate_1,'b', lw=2)
    ax.axhline(np.mean(S),color='k', ls=':', lw=2)
    ax.set_ylabel('stimulus strength')
    sns.despine(ax=ax)
    
    ax = plt.subplot(3,1,2)
    ax.plot((S-rate_1)**2, 'r')
    ax.plot(rate_2, color='k')
    ax.plot(rate_3, color='b')
    ax.set_ylabel('stimulus variance')
    sns.despine(ax=ax)
    
    ax = plt.subplot(3,1,3)
    ax.plot(S, 'r')
    ax.plot(rate_1,'b', lw=1)
    ax.plot(out,'k', lw=2)
    ax.set_ylabel('weighted output')
    ax.set_xlabel('stimulus number')
    sns.despine(ax=ax)
    