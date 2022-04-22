#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:30:15 2022

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
from src.Functions_MeanFieldNet import RunStaticNetwork_MFN

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Systematically vary key parameters

flag = 0

if flag==1:
    
    ### files and folders
    folder = 'Certainty_sensory_input'
    folder_pre = 'Results/Data/'
    fln_load_after = 'Data_Optimal_Parameters_MFN_10'
    fln_save_data = 'Example_mean_field'

    with open(folder_pre + folder + '/' + fln_load_after + '.pickle','rb') as f:
        xopt, W, optimize_flag, _, _ = pickle.load(f)
   
    W[optimize_flag!=0] = xopt
    stimuli = np.random.uniform(3, 9, size=50)
    
    # para_tested = np.array([0,0.5,1]) # BL
    # para_tested = np.array([20.0,40.0,60.0]) # tau_E
    # para_tested = np.array([2.,10.,50.]) # tau_I
    # para_tested = np.array([0.01,0.1,1]) # for SD, stimuli_std
    # para_tested = np.array([100,200,500,1000]) # stim_duration
    # para_tested = np.array([1000,4000,8000]) # tau_smooth
    para_tested = np.array([1e3, 1e4, 1e5]) # tau_output
    
    fig = plt.figure()#, tight_layout=True)
    ax = plt.subplot(1,1,1)
    
    for i in range(len(para_tested)):
    
        r_target = dtype([0, 0, 0, 0, 4, 4, 4, 4]) #dtype([para_tested[i], para_tested[i], 0, 0, 4, 4, 4, 4]) # dtype([0, 0, 0, 0, 4, 4, 4, 4])
        fixed_input = (np.eye(8) - W) @ r_target
        tau_inv = [dtype(1/60), dtype(1.0/2.0)] # [dtype(1/60), dtype(1/para_tested[i])] # [dtype(1/para_tested[i]), dtype(1.0/2.0)] # [dtype(1/60), dtype(1.0/2.0)]
        
        ### Simulation parameters
        dt = dtype(1) # 0.1 
        stim_duration = dtype(1000) # dtype(para_tested[i]) # dtype(1000)
        
        ### Stimulation protocol
        SD_individual = dtype(1e-5)  # dtype(1e-5) #dtype(para_tested[i])
        stimuli_std = np.ones_like(stimuli) * dtype(1e-5) # dtype(para_tested[i]) # dtype(1e-5)
        
        ### Run static network - MFN
        tau_smooth = dtype(8000) # dtype(para_tested[i]) # dtype(8000)
        RunStaticNetwork_MFN(W, tau_inv, tau_smooth, stim_duration, dt, fixed_input, 
                             stimuli, stimuli_std, SD_individual, folder, fln_save_data)
        
        # ### Load PE activity
        PathData = 'Results/Data/' + folder
        arr = np.loadtxt(PathData + '/Data_StaticNetwork_MFN_' + fln_save_data + '.dat', delimiter=' ')
        t, RE = arr[:,0], arr[:, 1:3]
        
        N_PE = np.size(RE,1)
        out_sum_PE = np.sum(RE,1)
        out_sum_PE2 = np.sum(RE**2,1)
        
        ### output neuron
        tc = dtype(para_tested[i]) # dtype(1e4)
        out_variance = np.zeros(len(t))
        out_variance[0] = out_sum_PE2[0]/tc
        
        for step in range(1,len(t)):
              out_variance[step] = (1-1/tc) * out_variance[step-1] + out_sum_PE2[step]/tc 
            
        # plot
        ax.plot(np.arange(len(out_sum_PE2))/stim_duration, (np.var(stimuli)-out_variance)**2, lw=2, label=str(para_tested[i]))
    
    ax.legend(loc=0)
    ax.axhline(0, color='k', ls=':')
    ax.set_xlabel('stimuli number'), ax.set_ylabel('MSE')
    sns.despine(ax=ax)


# %% Mean-field implementation (speed-up and check hypothesis for discrepancy, see below)

flag = 0

if flag==1:
    
    ### files and folders
    folder = 'Certainty_sensory_input'
    folder_pre = 'Results/Data/'
    fln_load_after = 'Data_Optimal_Parameters_MFN_10'
    fln_save_data = 'Example_mean_field'

    with open(folder_pre + folder + '/' + fln_load_after + '.pickle','rb') as f:
        xopt, W, optimize_flag, _, _ = pickle.load(f)
   
    W[optimize_flag!=0] = xopt
    r_target = dtype([0, 0, 0, 0, 4, 4, 4, 4])
    fixed_input = (np.eye(8) - W) @ r_target
    tau_inv = [dtype(1/60), dtype(1.0/2.0)]
    
    ### Simulation parameters
    dt = dtype(1) # 0.1 
    stim_duration = dtype(1000)
    
    ### Stimulation protocol
    SD_individual = dtype(1e-5)
    
    stimuli_first_trial = np.random.uniform(0, 10, size=50)
    stimuli_second_trial = np.random.uniform(3, 7, size=50)
    stimuli = np.concatenate((stimuli_first_trial,stimuli_second_trial))
    stimuli_std = 1e-5 * np.ones_like(stimuli)
    
    ### Run static network - MFN
    tau_smooth = 8000
    RunStaticNetwork_MFN(W, tau_inv, tau_smooth, stim_duration, dt, fixed_input, 
                         stimuli, stimuli_std, SD_individual, folder, fln_save_data)
    
    # ### Load PE activity
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticNetwork_MFN_' + fln_save_data + '.dat', delimiter=' ')
    t, RE = arr[:,0], arr[:, 1:3]
    
    N_PE = np.size(RE,1)
    out_sum_PE = np.sum(RE,1)
    out_sum_PE2 = np.sum(RE**2,1)
    
    ### output neuron
    tc = 1e4
    out_variance = np.zeros(len(t))
    out_variance[0] = out_sum_PE2[0]/tc
    
    for step in range(1,len(t)):
          out_variance[step] = (1-1/tc) * out_variance[step-1] + out_sum_PE2[step]/tc 
        
    # plot
    fig = plt.figure(figsize=(10,8))#, tight_layout=True)
    fig.subplots_adjust(hspace=0.5)
    
    ax = plt.subplot(2,1,1)
    ax.plot(stimuli, color='#8E4162', lw=2)
    ax.axhline(np.mean(stimuli), ls=':', color='k')
    ax.axvline(50, ls='--', color='k')
    ax.set_ylabel('stimulus strength')
    sns.despine(ax=ax)
    
    ax = plt.subplot(2,1,2)
    ax.plot(np.arange(len(out_sum_PE2))/stim_duration, out_sum_PE2, alpha=0.5, lw=2)
    ax.plot(np.arange(len(out_sum_PE2))/stim_duration, out_variance,'r', lw=2)
    ax.axhline(np.var(stimuli_first_trial), 0, 0.5)
    ax.axhline(np.var(stimuli_second_trial), 0.5)
    ax.axvline(50, ls='--', color='k')
    ax.set_xlabel('stimuli number'), ax.set_ylabel('activity')
    ax.set_title('PE circuit')
    sns.despine(ax=ax)


# %% First test

# mistake, see comment at line computing  out_sum_PE2

flag = 0

if flag==1:

    ### files and folders
    folder = 'Certainty_sensory_input'
    folder_pre = 'Results/Data/'
    fln_load_after = 'Example_Target_Input_After'
    fln_save_data = 'XXX'
    
      
    ### Load data
    with open(folder_pre + folder + '/Data_NetworkParameters_' + fln_load_after + '.pickle','rb') as f: 
        NeuPar, NetPar, InPar, _, _, RatePar, _ = pickle.load(f)
        
    with open(folder_pre + folder + '/Activity_relative_to_BL_' + fln_load_after + '.pickle','rb') as f:
        _, _, _, _, _, bool_nPE, bool_pPE = pickle.load(f)
    
    ### Simulation parameters
    SimPar = Simulation(stim_duration=dtype(500), dt=dtype(1))
    
    ### Stimulation protocol
    SD_individual = dtype(1e-5)
    
    stimuli_first_trial = np.random.uniform(0, 10, size=50)
    stimuli_second_trial = np.random.uniform(3, 7, size=50)
    stimuli = np.concatenate((stimuli_first_trial,stimuli_second_trial))
    stimuli_std = 1e-5 * np.ones_like(stimuli)
    
    prediction = np.mean(stimuli) * np.ones_like(stimuli) # should be running mean actually
    
    StimPar = Stimulation_new(stimuli, stimuli_std, SD_individual, prediction=prediction)
    
    ### Run static network
    RunStaticNetwork_new(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder, fln_save_data)
    
    ### Load PE activity
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticNetwork_' + fln_save_data + '.dat', delimiter=' ')
    t, RE = arr[:,0], arr[:, 1:141]
    
    PE = RE[:,(bool_nPE+bool_pPE)==1]
    N_PE = np.size(PE,1)
    out_sum_PE = np.sum(PE/N_PE,1)
    out_sum_PE2 = np.sum((PE**2)/N_PE,1) # ATTENTION: Actually you have to divide the nPE by N_nPE and the pPE by N_pPE (not by number of PE in total), otherwise it does not make sense and you cannot compare it to the toy model (see below)
    
    ### output neuron
    tc = 10
    out_variance = np.zeros(len(t))
    out_variance[0] = out_sum_PE2[0]/tc
    
    for step in range(1,len(t)):
         out_variance[step] = (1-1/tc) * out_variance[step-1] + out_sum_PE2[step]/tc
         #out_variance[step] = (1-1/tc) * out_variance[step-1] + out_sum_PE[step]**2/tc
     
        
    ### compare to toy model    
    SD = 0
    rate = np.zeros_like(stimuli)
    PE_activity = np.zeros_like(stimuli)
    
    for i, s in enumerate(stimuli):
            
        pPE = (np.maximum(s - prediction[i],0))**2
        nPE = (np.maximum(prediction[i] - s,0))**2
        SD = (1-1/tc) * SD + (nPE+pPE)/tc
            
        rate[i] = SD
        PE_activity[i] = (nPE+pPE)
        
    # plot
    fig = plt.figure(figsize=(10,8))#, tight_layout=True)
    fig.subplots_adjust(hspace=0.5)
    
    ax = plt.subplot(3,1,1)
    ax.plot(stimuli, color='#8E4162', lw=2)
    ax.axhline(np.mean(stimuli), ls=':', color='k')
    ax.axvline(50, ls='--', color='k')
    ax.set_ylabel('stimulus strength')
    sns.despine(ax=ax)
    
    ax = plt.subplot(3,1,2)
    ax.plot(PE_activity, alpha=0.5, lw=2)
    ax.plot(rate, 'r', lw=2)
    ax.axhline(np.var(stimuli_first_trial),0, 0.5)
    ax.axhline(np.var(stimuli_second_trial),0.5)
    ax.axvline(50, ls='--', color='k')
    ax.set_ylabel('activity')
    ax.set_title('toy model')
    sns.despine(ax=ax)
    
    ax = plt.subplot(3,1,3)
    ax.plot(out_sum_PE2, alpha=0.5, lw=2)
    ax.plot(out_variance,'r', lw=2)
    ax.axhline(np.var(stimuli_first_trial),0, 0.5)
    ax.axhline(np.var(stimuli_second_trial),0.5)
    ax.axvline(50, ls='--', color='k')
    ax.set_xlabel('stimuli number'), ax.set_ylabel('activity')
    ax.set_title('PE circuit')
    sns.despine(ax=ax)
    
    #plt.subplot_tool()
    
# %% Test "current" variance  - simple toy model

flag = 0

if flag==1:

    S1 = np.random.uniform(0, 10, size=1000)
    S2 = np.random.uniform(3, 7, size=1000)
    S = np.concatenate((S1,S2))
    
    tau = 100
    tau_SD = 50
    E = 0
    SD = 0
    
    rate = np.zeros_like(S)
    pred = np.zeros_like(S)
    
    for i, s in enumerate(S):
        
        #for t in np.arange(1000):
            
        pPE = (np.maximum(s - E,0))**2
        nPE = (np.maximum(E - s,0))**2
        E = (1-1/tau) * E + s/tau
        SD = (1-1/tau_SD) * SD + (nPE+pPE)/tau_SD
            
        rate[i] = SD
        pred [i] = E
        
    plt.figure()
    plt.plot(rate)
    plt.plot(pred)
