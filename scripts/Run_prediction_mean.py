#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:35:16 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src.Default_values import Default_PredProcPara
#from src.Functions_Analysis import Output_PredNet, Output_per_stimulus_PredNet, Output_ModLearnRate
from src.Functions_PredNet import RunPredNet, Neurons, Network, InputStructure, Stimulation, Simulation, Activity_Zero
from src.Functions_PredNet import RunPredNet_MFN #Network_MFN, InputStructure_MFN

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Universal settings

color_sensory_input = '#F3A712'
color_running_mean = '#5C7457'
color_prediction = '#3E517A'
color_pPE = '#A92B0F'
color_nPE = '#541388'
color_stim_duration = '#D59AC6'

# %%

# TODO: Show how MSE changes with increasing distribution std (or other moments)

# %% Impact of distribution type

# I would need to add bimodal distributions, and maybe Markov chains ...
# Also, I guess I have to run it with more seeds or vary stim_duration?
# Can I find differences?
#### Attention: Should I use running mean for MSE or should I use mean of distribution? Does it make a huge difference?

flag = 0
flg_dist = 3

if flag==1:
    
    ### files and folders
    folder = 'Prediction'
    folder_pre = 'Results/Data/'
    fln_load_after = 'Data_Optimal_Parameters_MFN_10'
    fln_save_data = 'Temporary'
    
    ### run parameters
    num_repeats = np.int32(50)
    seeds = np.arange(num_repeats, dtype=np.int32())
    n_stimuli = np.int32(50)
    
    ### distribution specifics
    mean_distribution = dtype(5)
    sd_distribution = dtype(2)
    
    ### function
    def run_single_network_with_predefined_stimuli(stimuli, folder, fln_load_after, fln_save_data, mLE=5e-3):
        
        ### network parameterisation
    
        ## Load data and set default values    
        with open(folder_pre + folder + '/' + fln_load_after + '.pickle','rb') as f:
            xopt, W, optimize_flag, _, _ = pickle.load(f)
       
        W[optimize_flag!=0] = xopt
        r_target = dtype([0, 0, 0, 0, 4, 4, 4, 4])
        fixed_input = (np.eye(8) - W) @ r_target
        tau_inv = [dtype(1/60), dtype(1.0/2.0)]
        
        ## Simulation parameters
        dt = dtype(1) # 0.1 
        stim_duration = dtype(500)
        
        ## Define connectivity between PE circuit and L
        U = np.zeros((1,8))     # PE circuit --> L
        U[0,0] = -mLE           # nPE onto L
        U[0,1] = mLE            # pPE onto L
        
        V = np.zeros((8,1))     # L --> PE circuit
        V[2:4,0] = dtype(1)     # onto dendrites
        V[5,0] = dtype(1)       # onto PV neuron receiving prediction
        V[7,0] = dtype(1)       # onto V neuron
        
        ## Stimulation protocol & Inputs
        SD = dtype(0.01)
        stimuli_std = np.ones_like(stimuli) * dtype(1)
        
        ### Run network
        RunPredNet_MFN(U, V, W, tau_inv, stim_duration, dt, fixed_input, stimuli, stimuli_std, SD, folder, fln_save_data)

        ### Compute mean-squared deviation between prediction and running mean
        PathData = 'Results/Data/' + folder
        arr = np.loadtxt(PathData + '/Data_StaticNetwork_MFN_' + fln_save_data + '.dat',delimiter=' ')
        t, R, L = arr[:,0], arr[:, 1:-2], arr[:,-1]
        
        MSE = (np.cumsum(stimuli)/np.arange(1,len(stimuli)+1) - L[::int(stim_duration)])**2
        
        return MSE
    
    ### initialise
    MSE = np.zeros((len(seeds),n_stimuli))
    
    for id_seed, seed in enumerate(seeds):
    
        np.random.seed(seed)
        
        ### create stimuli
        if flg_dist==0:
            stimuli = np.random.normal(mean_distribution, sd_distribution, size=n_stimuli)
            fln = 'normal'
        elif flg_dist==1:
            b = np.sqrt(12) * sd_distribution / 2 + mean_distribution
            a = 2 * mean_distribution -b
            stimuli = np.random.uniform(a, b, size=n_stimuli)
            fln = 'uniform'
        elif flg_dist==2:
            a = np.log(mean_distribution**2/np.sqrt(mean_distribution**2 + sd_distribution**2))
            b = np.sqrt(np.log(sd_distribution**2 / mean_distribution**2 + 1))
            stimuli = np.random.lognormal(a, b, size=n_stimuli)
            fln = 'log-normal'
        elif flg_dist==3:
            shape = mean_distribution**2 / sd_distribution**2
            scale = sd_distribution**2  / mean_distribution 
            stimuli = np.random.gamma(shape, scale, size=n_stimuli)
            fln = 'gamma'
    
        MSE[id_seed, :] = run_single_network_with_predefined_stimuli(stimuli, folder, fln_load_after, fln_save_data)
    
    ### save data
    filename = folder_pre + folder + '/Data_distribution_' + str(fln) + '.pickle'
    with open(filename,'wb') as f:
        pickle.dump([mean_distribution, sd_distribution, seeds, MSE],f)

    
# %% Figure: Impact of distribution type

flag = 1

if flag==1:
    
    plt.figure()
    ax = plt.gca()
    fln_dists = ['normal', 'uniform', 'log-normal', 'gamma']
    
    for dist_name in fln_dists:
        
        filename = folder_pre + folder + '/Data_distribution_' + str(dist_name) + '.pickle'
        with open(filename,'rb') as f:
            mean_distribution, sd_distribution, seeds, MSE = pickle.load(f)
            
        ax.plot(np.mean(MSE,0), label=dist_name)
        ax.fill_between(np.arange(np.size(MSE,1)), np.mean(MSE,0) - np.std(MSE,0)/np.sqrt(len(seeds)), np.mean(MSE,0) + np.std(MSE,0)/np.sqrt(len(seeds)), alpha=0.5)

    ax.legend(loc=0)
    sns.despine(ax=ax)

# %% Prediction equals the mean of the distribution only if nPE and pPE neurons exert symmetrical impact ... otherwise biased

# RUN !!! --> haven't done yet

flag = 0
flg = 0

if flag==1:
    
    ### run parameters
    num_repeats = np.int32(10)
    seeds = np.arange(num_repeats, dtype=np.int32())
    n_stimuli = np.int32(100)
    
    weight_mLE = np.array([5e-3, 1e-2, 2e-2])
    BL = np.array([0, 1, 2])
    
    ### files and folders
    folder = 'Prediction'
    folder_pre = 'Results/Data/'
    fln_load_after = 'Data_Optimal_Parameters_MFN_10'
    fln_save_data = 'Temporary'
    
    ### define function to set parameters and run network
    def run_single_network_return_MSE(n_stimuli, seed, folder, fln_load_after, fln_save_data, BL_nPE = 0, BL_pPE = 0, mLE_nPE = 5e-3, mLE_pPE = 5e-3):
        
        ### network parameterisation
    
        ## Load data and set default values    
        with open(folder_pre + folder + '/' + fln_load_after + '.pickle','rb') as f:
            xopt, W, optimize_flag, _, _ = pickle.load(f)
       
        W[optimize_flag!=0] = xopt
        tau_inv = [dtype(1/60), dtype(1.0/2.0)]
        
        ## Simulation parameters
        dt = dtype(1) # 0.1 
        stim_duration = dtype(500)
        
        ## Define connectivity between PE circuit and L
        U = np.zeros((1,8))     # PE circuit --> L
        U[0,0] = -mLE_nPE           # nPE onto L
        U[0,1] = mLE_pPE            # pPE onto L
        
        V = np.zeros((8,1))     # L --> PE circuit
        V[2:4,0] = dtype(1)     # onto dendrites
        V[5,0] = dtype(1)       # onto PV neuron receiving prediction
        V[7,0] = dtype(1)       # onto V neuron
        
        ## Stimulation protocol & Inputs
        mean_distribution = dtype(5)
        sd_distribution = dtype(2) 
        
        SD = dtype(0.01)
        n_stimuli = n_stimuli
        stimuli = np.zeros(n_stimuli, dtype=dtype)
        stimuli_std = np.ones_like(stimuli) * dtype(1)
        
        ### set random seed number
        np.random.seed(seed)
        
        ### targets and fixed inputs
        r_target = dtype([BL_nPE, BL_pPE, 0, 0, 4, 4, 4, 4])
        fixed_input = (np.eye(8) - W) @ r_target
        
        ### Stimulus distribution
        stimuli[:] = np.random.normal(mean_distribution, sd_distribution, size=int(n_stimuli))
        
        ### Run network
        RunPredNet_MFN(U, V, W, tau_inv, stim_duration, dt, fixed_input, stimuli, stimuli_std, SD, folder, fln_save_data)

        ### Compute mean-squared deviation between prediction and running mean
        PathData = 'Results/Data/' + folder
        arr = np.loadtxt(PathData + '/Data_StaticNetwork_MFN_' + fln_save_data + '.dat',delimiter=' ')
        t, R, L = arr[:,0], arr[:, 1:-2], arr[:,-1]
        
        MSE = (np.cumsum(stimuli)/np.arange(1,len(stimuli)+1) - L[::int(stim_duration)])**2
        
        return MSE
    

    if flg==0: ### weight from nPE to prediction neuron varied

        MSE = np.zeros((len(weight_mLE), len(seeds), n_stimuli))
        
        for id_para, para_tested in enumerate(weight_mLE):
            for id_seed, seed in enumerate(seeds):
                MSE[id_para, id_seed, :] = run_single_network_return_MSE(n_stimuli, seed, folder, fln_load_after, fln_save_data, mLE_nPE = para_tested)
                
        filename = folder_pre + folder + '/Data_vary_mLE_nPE.pickle'
        with open(filename,'wb') as f:
            pickle.dump([weight_mLE, seeds, MSE],f)
            
    elif flg==1:
        
        MSE = np.zeros((len(weight_mLE), len(seeds), n_stimuli))
        
        for id_para, para_tested in enumerate(weight_mLE):
            for id_seed, seed in enumerate(seeds):
                MSE[id_para, id_seed, :] = run_single_network_return_MSE(n_stimuli, seed, folder, fln_load_after, fln_save_data, mLE_pPE = para_tested)
                
        filename = folder_pre + folder + '/Data_vary_mLE_pPE.pickle'
        with open(filename,'wb') as f:
            pickle.dump([weight_mLE, seeds, MSE],f)
            
    elif flg==2:
        
        MSE = np.zeros((len(BL), len(seeds), n_stimuli))
        
        for id_para, para_tested in enumerate(BL):
            for id_seed, seed in enumerate(seeds):
                MSE[id_para, id_seed, :] = run_single_network_return_MSE(n_stimuli, seed, folder, fln_load_after, fln_save_data, BL_pPE = para_tested, BL_nPE = para_tested)
                
        filename = folder_pre + folder + '/Data_vary_BL.pickle'
        with open(filename,'wb') as f:
            pickle.dump([weight_mLE, seeds, MSE],f)
            
    elif flg==3:
        
        MSE = np.zeros((len(BL), len(seeds), n_stimuli))
        
        for id_para, para_tested in enumerate(BL):
            for id_seed, seed in enumerate(seeds):
                MSE[id_para, id_seed, :] = run_single_network_return_MSE(n_stimuli, seed, folder, fln_load_after, fln_save_data, BL_nPE = para_tested)
                
        filename = folder_pre + folder + '/Data_vary_B_nPEL.pickle'
        with open(filename,'wb') as f:
            pickle.dump([weight_mLE, seeds, MSE],f)
            
    elif flg==4:
    
        MSE = np.zeros((len(BL), len(seeds), n_stimuli))
        
        for id_para, para_tested in enumerate(BL):
            for id_seed, seed in enumerate(seeds):
                MSE[id_para, id_seed, :] = run_single_network_return_MSE(n_stimuli, seed, folder, fln_load_after, fln_save_data, BL_pPE = para_tested)
                
        filename = folder_pre + folder + '/Data_vary_B_nPEL.pickle'
        with open(filename,'wb') as f:
            pickle.dump([weight_mLE, seeds, MSE],f)
            

# %% Figure: Prediction equals the mean of the distribution only if nPE and pPE neurons exert symmetrical impact ... otherwise biased

flag = 0

if flag==1:
    
    print('to be done')
    

# %% PE neurons may establish a prediction of a random sensory input in form of the mean of the underlying stimulus distribution ...

flag = 0

if flag==1:
    
    ### files and folders
    folder = 'Prediction'
    folder_pre = 'Results/Data/'
    fln_load_after = 'Data_Optimal_Parameters_MFN_10'
    fln_save_data = 'Example_mean_field'
    
    ### network parameterisation
    
    ## Load data and set default values
    _, mLE, _, _, _, stimuli_weak, stimuli_strong = Default_PredProcPara()
    mLE *= 0.5

    with open(folder_pre + folder + '/' + fln_load_after + '.pickle','rb') as f:
        xopt, W, optimize_flag, _, _ = pickle.load(f)
   
    W[optimize_flag!=0] = xopt
    r_target = dtype([0, 0, 0, 0, 4, 4, 4, 4])
    fixed_input = (np.eye(8) - W) @ r_target
    tau_inv = [dtype(1/60), dtype(1.0/2.0)]
    
    ## Simulation parameters
    dt = dtype(1) # 0.1 
    stim_duration = dtype(1000)
    
    ## Define connectivity between PE circuit and L
    U = np.zeros((1,8))     # PE circuit --> L
    U[0,0] = -mLE           # nPE onto L
    U[0,1] = mLE            # pPE onto L
    
    V = np.zeros((8,1))     # L --> PE circuit
    V[2:4,0] = dtype(1)     # onto dendrites
    V[5,0] = dtype(1)       # onto PV neuron receiving prediction
    V[7,0] = dtype(1)       # onto V neuron
    
    
    ### Stimulus distribution
    
    ## Stimulation protocol & Inputs
    mean_distribution = dtype(5)
    sd_distribution = dtype(2) 
    
    SD = dtype(0.01)
    n_stimuli = 100
    stimuli = np.zeros(n_stimuli, dtype=dtype)
    stimuli_std = np.ones_like(stimuli) * dtype(1)
    stimuli[:] = np.random.normal(mean_distribution, sd_distribution, size=int(n_stimuli))
    
    ### Run network
    RunPredNet_MFN(U, V, W, tau_inv, stim_duration, dt, fixed_input, stimuli, stimuli_std, SD, folder, fln_save_data)
    
    ### Save relevant data
    filename = folder_pre + folder + '/Para_' + fln_save_data + '.pickle'
    with open(filename,'wb') as f:
        pickle.dump([U, V, W, tau_inv, stim_duration, dt, fixed_input, stimuli, stimuli_std, SD, mean_distribution, sd_distribution],f) 
        
        
# %% ... rather independent of the stimulus duration (steady state doesn't need to be reached)
    
flag = 0

if flag==1:
    
    ### files and folders
    folder = 'Prediction'
    folder_pre = 'Results/Data/'
    fln_load_after = 'Data_Optimal_Parameters_MFN_10'
    fln_save_data = 'Data_vary_stim_duration'
    
    ### run parameters
    num_repeats = np.int32(10)
    seeds = np.arange(num_repeats, dtype=np.int32())
    stimulus_durations = np.array([100,200,500,1000], dtype=dtype)
    
    ### network parameterisation
    
    ## Load data and set default values
    _, mLE, _, _, _, stimuli_weak, stimuli_strong = Default_PredProcPara()
    mLE *= 0.5

    with open(folder_pre + folder + '/' + fln_load_after + '.pickle','rb') as f:
        xopt, W, optimize_flag, _, _ = pickle.load(f)
   
    W[optimize_flag!=0] = xopt
    r_target = dtype([0, 0, 0, 0, 4, 4, 4, 4])
    fixed_input = (np.eye(8) - W) @ r_target
    tau_inv = [dtype(1/60), dtype(1.0/2.0)]
    
    ## Simulation parameters
    dt = dtype(1) # 0.1 
    
    ## Define connectivity between PE circuit and L
    U = np.zeros((1,8))     # PE circuit --> L
    U[0,0] = -mLE           # nPE onto L
    U[0,1] = mLE            # pPE onto L
    
    V = np.zeros((8,1))     # L --> PE circuit
    V[2:4,0] = dtype(1)     # onto dendrites
    V[5,0] = dtype(1)       # onto PV neuron receiving prediction
    V[7,0] = dtype(1)       # onto V neuron
    
    ## Stimulation protocol & Inputs
    mean_distribution = dtype(5)
    sd_distribution = dtype(2) 
    
    SD = dtype(0.01)
    n_stimuli = 150
    stimuli = np.zeros(n_stimuli, dtype=dtype)
    stimuli_std = np.ones_like(stimuli) * dtype(1)
    
    ### initialise 
    MSE = np.zeros((len(stimulus_durations), len(seeds), n_stimuli))
    
    for id_stim, stim_duration in enumerate(stimulus_durations):
        
        print(id_stim)
        
        for id_seed, seed in enumerate(seeds):
            
            ### set random seed number
            np.random.seed(seed)
            
            ### Stimulus distribution
            stimuli[:] = np.random.normal(mean_distribution, sd_distribution, size=int(n_stimuli))
            
            ### Run network
            RunPredNet_MFN(U, V, W, tau_inv, stim_duration, dt, fixed_input, stimuli, stimuli_std, SD, folder, fln_save_data)
    
            ### Compute mean-squared deviation between prediction and running mean
            PathData = 'Results/Data/' + folder
            arr = np.loadtxt(PathData + '/Data_StaticNetwork_MFN_' + fln_save_data + '.dat',delimiter=' ')
            t, R, L = arr[:,0], arr[:, 1:-2], arr[:,-1]
            
            MSE[id_stim, id_seed, :] = (np.cumsum(stimuli)/np.arange(1,len(stimuli)+1) - L[::int(stim_duration)])**2
    
    
    ### Save relevant data
    filename = folder_pre + folder + '/' + fln_save_data + '.pickle'
    with open(filename,'wb') as f:
        pickle.dump([U, V, W, tau_inv, stimulus_durations, dt, fixed_input, stimuli, stimuli_std, SD, mean_distribution, sd_distribution, seeds, MSE],f) 
    
    
# %% Figure: PE neurons may establish a prediction of a random sensory input in form of the mean of the underlying stimulus distribution

flag = 0

if flag==1:
    
    ## files and folders
    folder = 'Prediction'
    folder_pre = 'Results/Data/'
    fln_run = 'Example_mean_field'
    fln_duration = 'Data_vary_stim_duration'
    
    ### load data
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticNetwork_MFN_' + fln_run + '.dat',delimiter=' ')
    t, R, L = arr[:,0], arr[:, 1:-2], arr[:,-1]
    
    filename = folder_pre + folder + '/Para_' + fln_run + '.pickle'
    with open(filename,'rb') as f:
        _, _, _, _, stim_duration, _, _, stimuli, _, _, mean_distribution, sd_distribution = pickle.load(f) 
        
    filename = folder_pre + folder + '/' + fln_duration + '.pickle'
    with open(filename,'rb') as f:
        _, _, _, _, stimulus_durations, _, _, _, _, _, _, _, seeds, MSE = pickle.load(f) 
    
    ### figure settings
    fig = plt.figure(figsize = (15,5), tight_layout=True)
    gs = gridspec.GridSpec(ncols=5, nrows=4, figure=fig)
    
    ### input distribution
    ax = fig.add_subplot(gs[:2, 0])
    example_distribution = np.random.normal(mean_distribution, sd_distribution, size=int(1e5))
    ax.hist(example_distribution, 20, density=True, color=color_sensory_input)
    ax.axvline(mean_distribution, color=color_running_mean, lw=3)
    ax.axvline(mean_distribution-sd_distribution, color='k', ls='--', lw=2)
    ax.axvline(mean_distribution+sd_distribution, color='k', ls='--', lw=2)
    ax.set_xlabel('sensory input')
    ax.set_ylabel('probability density')
    sns.despine(ax=ax)
    
    ### prediction & running mean
    ax = fig.add_subplot(gs[:3, 1:3])
    ax.plot(np.arange(len(stimuli)), stimuli, 'o', color=color_sensory_input, alpha=0.5)
    ax.plot(np.arange(len(stimuli)), np.cumsum(stimuli)/np.arange(1,len(stimuli)+1), color=color_running_mean, ls='-', lw=3)
    ax.plot(t/stim_duration, L, color=color_prediction, lw=3)
    ax.set_ylabel('Prediction')
    #ax.set_xlabel('trials')
    sns.despine(ax=ax)
    
    ### PE neuron activity
    ax = fig.add_subplot(gs[3:, 1:3])
    ax.plot(t/stim_duration, R[:,0], color=color_nPE)
    ax.plot(t/stim_duration, R[:,1], color=color_pPE)
    ax.set_ylabel('Deviation')
    ax.set_xlabel('trials')
    #ax.legend(['dev', 'nPE', 'pPE'], ncol=3)
    sns.despine(ax=ax)
    
    ### this is independent of stimulus duration (but the shorter the stimuli the longer it takes to establish the mean)
    ax = fig.add_subplot(gs[:3, 3:])
    
    color_palette = sns.dark_palette(color_stim_duration, reverse=True, n_colors=len(stimulus_durations))
    MSE_averaged_over_seeds = np.mean(MSE,axis=1)
    MSE_SEM_over_seeds = np.std(MSE,axis=1)/np.sqrt(len(seeds))
    n_stimuli = np.size(MSE,2)
    
    for num_para, stim_duration in enumerate(stimulus_durations):
        ax.plot(MSE_averaged_over_seeds[num_para,:], color=color_palette[num_para])
        ax.fill_between(np.arange(n_stimuli), MSE_averaged_over_seeds[num_para,:]-MSE_SEM_over_seeds[num_para,:], 
                        MSE_averaged_over_seeds[num_para,:]+MSE_SEM_over_seeds[num_para,:], color=color_palette[num_para], alpha=0.5)
    
    ax.set_ylabel('MSE')
    ax.set_xlabel('trials')
    sns.despine(ax=ax)


# %% ##########################################################################
############################# Before 20/04/22 #################################
###############################################################################

# %% Look at different distributions 

flag = 0
flg_dist = 4

if flag==1:

    ### files and folders
    folder = 'Prediction'
    folder_pre = 'Results/Data/'
    fln_load_after = 'Data_Optimal_Parameters_MFN_10'
    fln_save_data = 'Example_mean_field'
    
    ### Load data and set default values
    _, mLE, _, _, _, stimuli_weak, stimuli_strong = Default_PredProcPara()
    mLE *= 0.5

    with open(folder_pre + folder + '/' + fln_load_after + '.pickle','rb') as f:
        xopt, W, optimize_flag, _, _ = pickle.load(f)
   
    W[optimize_flag!=0] = xopt
    r_target = dtype([0, 0, 0, 0, 4, 4, 4, 4])
    fixed_input = (np.eye(8) - W) @ r_target
    tau_inv = [dtype(1/60), dtype(1.0/2.0)]
    
    ### Simulation parameters
    dt = dtype(1) # 0.1 
    stim_duration = dtype(1000)
    
    ### Stimulation protocol & Inputs
    SD = dtype(0.01)
    n_stimuli = 100
    stimuli = np.zeros(n_stimuli, dtype=dtype)
    stimuli_std = np.ones_like(stimuli) * dtype(1)
    if flg_dist==0: # uniform
        stimuli[:] = np.random.uniform(1, 5 , size=int(n_stimuli))
    elif flg_dist==1:# normal
        stimuli[:] = np.random.normal(3, 1.15 , size=int(n_stimuli))
    elif flg_dist==2:# log-normal
        stimuli[:] = np.random.lognormal(1.04, 0.36, size=int(n_stimuli))
    elif flg_dist==3: # binary distribution
        stimuli[:] = np.random.choice([2,4], size=int(n_stimuli), p=[0.5,0.5])
    elif flg_dist==4:# 2 normal distributions
        stimuli[0::2] = np.random.normal(2, 0.5 , size=int(n_stimuli/2))
        stimuli[1::2] = np.random.normal(4, 0.5 , size=int(n_stimuli/2))
    
    ### Define connectivity between PE circuit and L
    U = np.zeros((1,8)) # PE circuit --> L
    U[0,0] = -mLE # nPE onto L
    U[0,1] = mLE # pPE onto L
    
    V = np.zeros((8,1)) # L --> PE circuit
    V[2:4,0] = dtype(1) # onto dendrites
    V[5,0] = dtype(1) # onto PV neuron receiving prediction
    V[7,0] = dtype(1) # onto V neuron
    
    ### Run network
    RunPredNet_MFN(U, V, W, tau_inv, stim_duration, dt, fixed_input, stimuli, stimuli_std, SD, folder, fln_save_data)
    
    ### Plotting
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticNetwork_MFN_' + fln_save_data + '.dat',delimiter=' ')
    t, R, L = arr[:,0], arr[:, 1:-2], arr[:,-1]
    
    plt.figure(figsize = (10,10), tight_layout=True)
    ax = plt.subplot(3,2,1)
    ax.hist(stimuli, color='#8E4162', bins=10, density=True)
    ax.axvline(np.mean(stimuli), color='#738C54', lw=3)
    ax.set_ylabel('density'), ax.set_xlabel('mean of trial stimulus')
    ax.set_title('Distribution of trial mean')
    sns.despine(ax=ax)
    
    ax = plt.subplot(3,2,2)
    num_time_steps = np.int32(stim_duration/dt) 
    example_trial_1 = np.random.normal(stimuli[0],stimuli_std[0], size=num_time_steps)
    example_trial_2 = np.random.normal(stimuli[1],stimuli_std[1], size=num_time_steps)
    example_trial = np.concatenate((example_trial_1, example_trial_2))
    ax.plot(example_trial, color='#8E4162')
    ax.axvline(num_time_steps, lw=3, color='k')
    ax.set_title('Inputs during the first two trials')
    ax.set_xlabel('Time'), ax.set_ylabel('stimulus')
    sns.despine(ax=ax)
    
    ax = plt.subplot(3,2,(3,4))
    ax.plot(np.arange(len(stimuli)), stimuli, 'o', color='#8E4162', alpha=0.5)
    ax.plot(np.arange(len(stimuli)), np.cumsum(stimuli)/np.arange(1,len(stimuli)+1), color='#738C54', ls='-', lw=3)
    #ax.axhline(np.mean(stimuli), color='#738C54', ls='-', lw=3)
    ax.plot(t/stim_duration, L, color='#065A82', lw=3)
    ax.set_ylabel('Prediction')
    ax.set_xlabel('trials')
    sns.despine(ax=ax)
    
    ax = plt.subplot(3,2,(5,6))
    ax.plot(np.abs(stimuli - L[::int(stim_duration)]), 'k', marker='_', linestyle="None")
    ax.plot(t/stim_duration, R[:,0])
    ax.plot(t/stim_duration, R[:,1])
    ax.set_ylabel('Deviation')
    ax.set_xlabel('trials')
    ax.legend(['dev', 'nPE', 'pPE'], ncol=3)
    sns.despine(ax=ax)

# %% Systematically vary neural and network parameters to study their influence on the networks ability to predict the mean

flag = 0

if flag==1:
    
    ### files and folders
    folder = 'Prediction'
    folder_pre = 'Results/Data/'
    fln_load_after = 'Data_Optimal_Parameters_MFN_10'
    fln_save_data = 'Example_mean_field'
    
    ### Load data and set default values
    _, mLE, _, _, _, stimuli_weak, stimuli_strong = Default_PredProcPara()
    mLE *= 0.5

    with open(folder_pre + folder + '/' + fln_load_after + '.pickle','rb') as f:
        xopt, W, optimize_flag, _, _ = pickle.load(f)
   
    W[optimize_flag!=0] = xopt
    
    plt.figure(figsize = (5,3), tight_layout=True)
    ax = plt.subplot(1,1,1)
    
    # para_tested = np.array([0.01,0.1,1]) # for SD, stimuli_std
    # para_tested = np.array([20.0,40.0,60.0]) # tau_E
    # para_tested = np.array([2.,10.,50.]) # tau_I
    # para_tested = np.array([0,2,4]) # BL
    # para_tested = np.array([100,200,500,1000]) # stim_duration
    para_tested = np.array([1,2,4]) # factor multiply mLE (either nPE or pPE)
    
    n_stimuli = 100
    stimuli = np.zeros(n_stimuli, dtype=dtype)
    stimuli[:] = np.random.uniform(1, 5, size=int(n_stimuli))
    
    for i in range(len(para_tested)):
        
        r_target = dtype([0, 0, 0, 0, 4, 4, 4, 4])  # dtype([0, 0, 0, 0, 4, 4, 4, 4]) type([para_tested[i], para_tested[i], 0, 0, 4, 4, 4, 4])
        fixed_input = (np.eye(8) - W) @ r_target
        tau_inv = [dtype(1/60), dtype(1/2)] # [dtype(1/60), dtype(1.0/2.0)] [dtype(1/60), dtype(1/para_tested[i])] 
        
        ### Simulation parameters
        dt = dtype(1) # 0.1 
        stim_duration = dtype(1000) # dtype(para_tested[i]) # dtype(1000)
        
        ### noise parameters
        SD = dtype(0.01) # dtype(0.01) #dtype(para_tested[i])
        stimuli_std = np.ones_like(stimuli) * dtype(0.01) # dtype(para_tested[i]) # dtype(1)
        
        ### Define connectivity between PE circuit and L
        U = np.zeros((1,8)) # PE circuit --> L
        U[0,0] = -mLE # nPE onto L
        U[0,1] = mLE * para_tested[i] # pPE onto L
        
        V = np.zeros((8,1)) # L --> PE circuit
        V[2:4,0] = dtype(1) # onto dendrites
        V[5,0] = dtype(1) # onto PV neuron receiving prediction
        V[7,0] = dtype(1) # onto V neuron
        
        ### Run network
        RunPredNet_MFN(U, V, W, tau_inv, stim_duration, dt, fixed_input, stimuli, stimuli_std, SD, folder, fln_save_data)
        
        ### Plotting
        PathData = 'Results/Data/' + folder
        arr = np.loadtxt(PathData + '/Data_StaticNetwork_MFN_' + fln_save_data + '.dat',delimiter=' ')
        t, R, L = arr[:,0], arr[:, 1:-2], arr[:,-1]
        
        ax.plot(np.cumsum(stimuli)/np.arange(1,len(stimuli)+1) - L[::int(stim_duration)], '.-', label=str(para_tested[i]))
        ax.set_ylabel('Deviation')
        ax.set_xlabel('trials')
    
    ax.axhline(0,ls=':', color='k')
    ax.legend(loc=0)
    sns.despine(ax=ax)


# %% Use mean-field PE circuits (to speed up process and verify hypothesis -- see below)

flag = 0

if flag==1:

    ### files and folders
    folder = 'Prediction'
    folder_pre = 'Results/Data/'
    fln_load_after = 'Data_Optimal_Parameters_MFN_10'
    fln_save_data = 'Example_mean_field'
    
    ### Load data and set default values
    _, mLE, _, _, _, stimuli_weak, stimuli_strong = Default_PredProcPara()
    mLE *= 0.5

    with open(folder_pre + folder + '/' + fln_load_after + '.pickle','rb') as f:
        xopt, W, optimize_flag, _, _ = pickle.load(f)
   
    W[optimize_flag!=0] = xopt
    r_target = dtype([0, 0, 0, 0, 4, 4, 4, 4])
    fixed_input = (np.eye(8) - W) @ r_target
    tau_inv = [dtype(1/60), dtype(1.0/2.0)]
    
    ### Simulation parameters
    dt = dtype(1) # 0.1 
    stim_duration = dtype(1000)
    
    ### Stimulation protocol & Inputs
    SD = dtype(0.01)
    n_stimuli = 100
    stimuli = np.zeros(n_stimuli, dtype=dtype)
    stimuli_std = np.ones_like(stimuli) * dtype(1)
    stimuli[:] = np.random.uniform(1, 5, size=int(n_stimuli))
    
    ### Define connectivity between PE circuit and L
    U = np.zeros((1,8)) # PE circuit --> L
    U[0,0] = -mLE # nPE onto L
    U[0,1] = mLE # pPE onto L
    
    V = np.zeros((8,1)) # L --> PE circuit
    V[2:4,0] = dtype(1) # onto dendrites
    V[5,0] = dtype(1) # onto PV neuron receiving prediction
    V[7,0] = dtype(1) # onto V neuron
    
    ### Run network
    RunPredNet_MFN(U, V, W, tau_inv, stim_duration, dt, fixed_input, stimuli, stimuli_std, SD, folder, fln_save_data)
    
    ### Plotting
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticNetwork_MFN_' + fln_save_data + '.dat',delimiter=' ')
    t, R, L = arr[:,0], arr[:, 1:-2], arr[:,-1]
    
    plt.figure(figsize = (10,10), tight_layout=True)
    ax = plt.subplot(3,2,1)
    ax.hist(np.random.uniform(min(stimuli),max(stimuli),size=2000), color='#8E4162', bins=30, density=True)
    ax.axvline(np.mean(stimuli), color='#738C54', lw=3)
    ax.set_ylabel('density'), ax.set_xlabel('mean of trial stimulus')
    ax.set_title('Distribution of trial mean')
    sns.despine(ax=ax)
    
    ax = plt.subplot(3,2,2)
    num_time_steps = np.int32(stim_duration/dt) 
    example_trial_1 = np.random.normal(stimuli[0],stimuli_std[0], size=num_time_steps)
    example_trial_2 = np.random.normal(stimuli[1],stimuli_std[1], size=num_time_steps)
    example_trial = np.concatenate((example_trial_1, example_trial_2))
    ax.plot(example_trial, color='#8E4162')
    ax.axvline(num_time_steps, lw=3, color='k')
    ax.set_title('Inputs during the first two trials')
    ax.set_xlabel('Time'), ax.set_ylabel('stimulus')
    sns.despine(ax=ax)
    
    ax = plt.subplot(3,2,(3,4))
    ax.plot(np.arange(len(stimuli)), stimuli, 'o', color='#8E4162', alpha=0.5)
    ax.plot(np.arange(len(stimuli)), np.cumsum(stimuli)/np.arange(1,len(stimuli)+1), color='#738C54', ls='-', lw=3)
    #ax.axhline(np.mean(stimuli), color='#738C54', ls='-', lw=3)
    ax.plot(t/stim_duration, L, color='#065A82', lw=3)
    ax.set_ylabel('Prediction')
    ax.set_xlabel('trials')
    sns.despine(ax=ax)
    
    ax = plt.subplot(3,2,(5,6))
    ax.plot(np.abs(stimuli - L[::int(stim_duration)]), 'k', marker='_', linestyle="None")
    ax.plot(t/stim_duration, R[:,0])
    ax.plot(t/stim_duration, R[:,1])
    ax.set_ylabel('Deviation')
    ax.set_xlabel('trials')
    ax.legend(['dev', 'nPE', 'pPE'], ncol=3)
    sns.despine(ax=ax)
    

# %% First test

# maybe use hdf to make faster?
# mean over stimuli should be shown as sliding mean (not just one line for mean over all stimuli)

flag = 0

if flag==1:

    ### files and folders
    folder = 'Prediction'
    folder_pre = 'Results/Data/'
    fln_load_after = 'Example_Target_Input_After'
    fln_save_data = 'XX'
    
    ### Load data and set default values
    _, mLE, _, _, _, stimuli_weak, stimuli_strong = Default_PredProcPara()
    
    with open(folder_pre + folder + '/Activity_relative_to_BL_' + fln_load_after + '.pickle','rb') as f:
        _, _, _, _, _, bool_nPE, bool_pPE = pickle.load(f)
    
    
    with open(folder_pre + folder + '/Data_NetworkParameters_' + fln_load_after + '.pickle','rb') as f: 
        NeuPar_PE, NetPar_PE, InPar_PE, _, _, _, _ = pickle.load(f)
    
    
    ### Neuron parameters
    NeuPar = Neurons()#tau_inv_E=dtype(1/30))
    
    ### Network parameters
    NetPar = Network(NeuPar, NetPar_PE, mLE, bool_nPE, bool_pPE, InPar_PE.neurons_motor)
    
    ## Initial activity levels
    RatePar = Activity_Zero(NeuPar, 0)
    RatePar.rL0 = np.array([0], dtype=dtype)
    
    ### Simulation parameters
    SimPar = Simulation()
    
    ### Stimulation protocol & Inputs
    InPar = InputStructure(NeuPar, InPar_PE)
    
    n_stimuli = 100
    stimuli = np.zeros(n_stimuli)
    std_arr = np.ones_like(stimuli) #* 0.01
    stimuli[:] = np.random.uniform(1,5,size=int(n_stimuli))
    #stimuli[:] = np.random.normal(3,1.137,size=int(n_stimuli))
    StimPar = Stimulation(stimuli, std_arr, SD_individual = dtype(0.01))
    
    ### Run network 
    RunPredNet(NeuPar, NetPar, InPar, StimPar, SimPar, RatePar, folder, fln_save_data)
    
    ### Plotting
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticPredNet_P_' + fln_save_data + '.dat',delimiter=' ')
    t, L = arr[:,0], arr[:, 1]
    
    plt.figure(figsize = (10,7), tight_layout=True)
    ax = plt.subplot(2,2,1)
    ax.hist(np.random.uniform(min(stimuli),max(stimuli),size=2000), color='#8E4162', bins=30, density=True)
    ax.axvline(np.mean(stimuli), color='#738C54', lw=3)
    ax.set_ylabel('density'), ax.set_xlabel('mean of trial stimulus')
    ax.set_title('Distribution of trial mean')
    sns.despine(ax=ax)
    
    ax = plt.subplot(2,2,2)
    num_time_steps = np.int32(SimPar.stim_duration/SimPar.dt) 
    example_trial_1 = np.random.normal(StimPar.stimuli[0],StimPar.stimuli_std[0], size=num_time_steps)
    example_trial_2 = np.random.normal(StimPar.stimuli[1],StimPar.stimuli_std[1], size=num_time_steps)
    example_trial = np.concatenate((example_trial_1, example_trial_2))
    ax.plot(example_trial, color='#8E4162')
    ax.axvline(num_time_steps, lw=3, color='k')
    ax.set_title('Inputs during the first two trials')
    ax.set_xlabel('Time'), ax.set_ylabel('stimulus')
    sns.despine(ax=ax)
    
    ax = plt.subplot(2,2,(3,4))
    ax.plot(t/SimPar.stim_duration, stimuli, 'o', color='#8E4162', alpha=0.5)
    ax.axhline(np.mean(stimuli), color='#738C54', ls='-', lw=3)
    ax.plot(t/SimPar.stim_duration, L, color='#065A82', lw=3)
    ax.set_ylabel('Prediction')
    ax.set_xlabel('trials')
    sns.despine(ax=ax)

    
### What are the factors that determine how much the prediction deviates from the true prediction? --> maybe go back to toy example and check first there ...
    # time constants?
    # weights from PE to L? (does not seem to be the case ...)
    # SD (simulation with 0.1 instead of 1 did not show significant differences)
    # distribution type?
   
    
# %% Simple toy model 
    
flag = 0

if flag==1:

    S = np.random.uniform(1, 5, size=100)
    
    eta = 1e-3
    tau = 10
    E = 0
    s_SD = 1
    
    rate = np.zeros_like(S)
    
    for i, s in enumerate(S):
        
        stim = np.random.normal(s, s_SD, size=1000)
        
        for t in np.arange(1000):
            
            PE = (stim[t] - E)
            # if PE<0:
            #     E += eta * PE/tau * 0.5
            # else:
            #     E += eta * PE/tau #* 0.5
            E += eta * PE/tau
            
        rate[i] = E
        
    plt.figure()
    plt.plot(rate)
    ax = plt.gca()
    ax.axhline(np.mean(S), ls=':')
    