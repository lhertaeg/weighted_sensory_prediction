#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.default_parameters import Neurons, Activity_Zero, Network, Stimulation
from src.functions_save import load_network_para
from src.functions_networks import run_population_net

from src.plot_data import plot_gain_factors

dtype = np.float32


# %% plot gain factors

run_cell = True

if run_cell:
    
    import seaborn as sns
    import pandas as pd
    
    with open('../results/data/population/data_population_network_para4corr_gain.pickle','rb') as f:
            [_, _, gain_factors_nPE_5, gain_factors_pPE_5, nPE_true, pPE_true] = pickle.load(f)


    gains_nPE = np.round(gain_factors_nPE_5,1)
    gains_nPE[~nPE_true] = np.nan
    
    gains_pPE = np.round(gain_factors_pPE_5,1)
    gains_pPE[~pPE_true] = np.nan
    
    inch = 2.54
    figsize=(10/inch,2/inch)
    
    plot_gain_factors(np.log(gains_nPE), np.log(gains_pPE), figsize=figsize)

    figure_name = 'Figure_gains.png'
    figPath = '../results/figures/final/'
    plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=False, dpi=600)
   

# %% Sparsity 

run_cell = False
plot_only = True

if run_cell:
    
    # filename to save relevant data
    file_data = '../results/data/population/data_population_sparsity.pickle'
    
    if not plot_only:
    
        # get data
        with open('../results/data/population/data_population_network_parameters.pickle','rb') as f:
            [neurons_visual, inp_ext_soma, inp_ext_dend, weight_name, Dict_w, Dict_t] = pickle.load(f)
        
        with open('../results/data/population/data_population_network_para4corr_gain.pickle','rb') as f:
            [_, _, gain_factors_nPE_5, gain_factors_pPE_5, nPE_true, pPE_true] = pickle.load(f)
        
        # conditions to be tested
        p_conns = np.linspace(0.6, 1, 5)
        num_seeds = 10
        
        # initialise arrays
        M_steady_state = np.zeros((len(p_conns), num_seeds))
        V_steady_state = np.zeros((len(p_conns), num_seeds))
        
        for cond_num, p_conn in enumerate(p_conns):
            
            # parametrisation neurons, network and rates
            NeuPar = Neurons()
            NetPar = Network(NeuPar, Dict_w, Dict_t, weight_name, neurons_visual, gain_factors_nPE = gain_factors_nPE_5, 
                             gain_factors_pPE = gain_factors_pPE_5, nPE_true = nPE_true, pPE_true = pPE_true, p_conn=p_conn)
            
            for seed in range(num_seeds):
                
                print('p_conn = ', p_conn, ', seed = ', seed)
                
                # parametrisation rates and stimulation
                RatePar = Activity_Zero(NeuPar)
                StimPar = Stimulation(5, 2, inp_ext_soma, inp_ext_dend, neurons_visual, trial_duration = np.int32(100000), 
                                      num_values_per_trial = np.int32(200), seed=seed) # 20000, 20
                            
                # run network
                run_population_net(NeuPar, NetPar, StimPar, RatePar, 0.1, folder='population', fln='test')
                
                # load simulation data
                PathData = '../results/data/population'
                arr = np.loadtxt(PathData + '/Data_PopulationNetwork_test.dat',delimiter=' ')
                t, R = arr[:,0], arr[:, 1:]
                
                ind_break = np.cumsum(NeuPar.NCells,dtype=np.int32)
                ind_break = np.concatenate([ind_break, np.array([340,341])])
                
                rE, rP, rS, rV, rD, r_mem, r_var = np.split(R, ind_break, axis=1)
                
                avg_start = len(r_mem)//2
                M_steady_state[cond_num,seed] = np.mean(r_mem[-avg_start:])
                V_steady_state[cond_num,seed] = np.mean(r_var[-avg_start:])
                
                print(np.mean(r_mem[-avg_start:]))
                print(np.mean(r_var[-avg_start:]))
        
        # save data
        with open(file_data,'wb') as f:
            pickle.dump([num_seeds, p_conns, M_steady_state, V_steady_state],f)
            
    else:
        
        # load data
        with open(file_data,'rb') as f:
            [num_seeds, p_conns, M_steady_state, V_steady_state] = pickle.load(f)
            
    ### plot
    _, ax = plt.subplots(1,1)
    
    M_avg = np.mean((M_steady_state - 5)/5 * 100, 1)
    V_avg = np.mean((V_steady_state - 4)/4 * 100, 1)
    
    M_sem = np.std((M_steady_state - 5)/5 * 100, 1)/np.sqrt(num_seeds)
    V_sem = np.std((V_steady_state - 4)/4 * 100, 1)/np.sqrt(num_seeds)
    
    ax.plot(p_conns, M_avg, 'b')
    ax.fill_between(p_conns, M_avg - M_sem/2, M_avg + M_sem/2, color='b', alpha=0.5)
    
    ax.plot(p_conns, V_avg, 'r')
    ax.fill_between(p_conns, V_avg - V_sem/2, V_avg + V_sem/2, color='r', alpha=0.5)

    ax.set_ylim([-15,10])
    ax.set_xlabel('Sparsity (connection prop.)')
    ax.set_ylabel('Deviation (%)')

# %% Correlated deviations  

run_cell = False
plot_only = False

if run_cell:
    
    # filename to save relevant data
    file_data = '../results/data/population/data_population_correlated_deviations.pickle'
    
    if not plot_only:
    
        # get data
        with open('../results/data/population/data_population_network_parameters.pickle','rb') as f:
            [neurons_visual, inp_ext_soma, inp_ext_dend, weight_name, Dict_w, Dict_t] = pickle.load(f)
        
        with open('../results/data/population/data_population_network_para4corr_gain.pickle','rb') as f:
            [_, _, gain_factors_nPE_5, gain_factors_pPE_5, nPE_true, pPE_true] = pickle.load(f)
        
        # conditions to be tested
        means = np.linspace(0.85, 1.1, 6)
        num_seeds = 10
        
        # initialise arrays
        M_steady_state = np.zeros((len(means), num_seeds))
        V_steady_state = np.zeros((len(means), num_seeds))
        
        for cond_num, mean in enumerate(means):
            for seed in range(num_seeds):
                
                print('mean = ', mean, ', seed = ', seed)
                
                # parametrisation neurons, network and rates
                NeuPar = Neurons()
                RatePar = Activity_Zero(NeuPar)
                
                NetPar = Network(NeuPar, Dict_w, Dict_t, weight_name, neurons_visual, gain_factors_nPE = gain_factors_nPE_5, 
                         gain_factors_pPE = gain_factors_pPE_5, nPE_true = nPE_true, pPE_true = pPE_true, mean=mean, std=0)
                
                StimPar = Stimulation(5, 2, inp_ext_soma, inp_ext_dend, neurons_visual, trial_duration = np.int32(100000), 
                                      num_values_per_trial = np.int32(200), seed=seed) # 20000, 20
                            
                # run network
                run_population_net(NeuPar, NetPar, StimPar, RatePar, 0.1, folder='population', fln='test')
                
                # load simulation data
                PathData = '../results/data/population'
                arr = np.loadtxt(PathData + '/Data_PopulationNetwork_test.dat',delimiter=' ')
                t, R = arr[:,0], arr[:, 1:]
                
                ind_break = np.cumsum(NeuPar.NCells,dtype=np.int32)
                ind_break = np.concatenate([ind_break, np.array([340,341])])
                
                rE, rP, rS, rV, rD, r_mem, r_var = np.split(R, ind_break, axis=1)
                
                avg_start = len(r_mem)//2
                M_steady_state[cond_num,seed] = np.mean(r_mem[-avg_start:])
                V_steady_state[cond_num,seed] = np.mean(r_var[-avg_start:])
        
        # save data
        with open(file_data,'wb') as f:
            pickle.dump([num_seeds, means, M_steady_state, V_steady_state],f)
            
    else:
        
        # load data
        with open(file_data,'rb') as f:
            [num_seeds, means, M_steady_state, V_steady_state] = pickle.load(f)
            
    ### plot
    _, ax = plt.subplots(1,1)
    
    M_avg = np.mean((M_steady_state - 5)/5 * 100, 1)
    V_avg = np.mean((V_steady_state - 4)/4 * 100, 1)
    
    M_sem = np.std((M_steady_state - 5)/5 * 100, 1)/np.sqrt(num_seeds)
    V_sem = np.std((V_steady_state - 4)/4 * 100, 1)/np.sqrt(num_seeds)
    
    ax.plot(means, M_avg, 'b')
    ax.fill_between(means, M_avg - M_sem/2, M_avg + M_sem/2, color='b', alpha=0.5)
    
    ax.plot(means, V_avg, 'r')
    ax.fill_between(means, V_avg - V_sem/2, V_avg + V_sem/2, color='r', alpha=0.5)

    #ax.set_ylim(bottom=0)

    ax.set_xlabel('Correlated deviation (mean)')
    ax.set_ylabel('Deviation (%)')
    
# %% Uncorrelated deviations  

run_cell = False
plot_only = False

if run_cell:
    
    # filename to save relevant data
    file_data = '../results/data/population/data_population_uncorrelated_deviations.pickle'
    
    if not plot_only:
    
        # get data
        with open('../results/data/population/data_population_network_parameters.pickle','rb') as f:
            [neurons_visual, inp_ext_soma, inp_ext_dend, weight_name, Dict_w, Dict_t] = pickle.load(f)
        
        with open('../results/data/population/data_population_network_para4corr_gain.pickle','rb') as f:
            [_, _, gain_factors_nPE_5, gain_factors_pPE_5, nPE_true, pPE_true] = pickle.load(f)
        
        # conditions to be tested
        SDs = np.linspace(0, 0.5, 6)
        num_seeds = 10
        
        # initialise arrays
        M_steady_state = np.zeros((len(SDs), num_seeds))
        V_steady_state = np.zeros((len(SDs), num_seeds))
        
        for cond_num, sd in enumerate(SDs):
            for seed in range(num_seeds):
                
                print('sd = ', sd, ', seed = ', seed)
                
                # parametrisation neurons, network and rates
                NeuPar = Neurons()
                RatePar = Activity_Zero(NeuPar)
                
                NetPar = Network(NeuPar, Dict_w, Dict_t, weight_name, neurons_visual, gain_factors_nPE = gain_factors_nPE_5, 
                         gain_factors_pPE = gain_factors_pPE_5, nPE_true = nPE_true, pPE_true = pPE_true, mean=1, std=sd)
                
                StimPar = Stimulation(5, 2, inp_ext_soma, inp_ext_dend, neurons_visual, trial_duration = np.int32(100000), 
                                      num_values_per_trial = np.int32(200), seed=seed) # 20000, 20
                            
                # run network
                run_population_net(NeuPar, NetPar, StimPar, RatePar, 0.1, folder='population', fln='test')
                
                # load simulation data
                PathData = '../results/data/population'
                arr = np.loadtxt(PathData + '/Data_PopulationNetwork_test.dat',delimiter=' ')
                t, R = arr[:,0], arr[:, 1:]
                
                ind_break = np.cumsum(NeuPar.NCells,dtype=np.int32)
                ind_break = np.concatenate([ind_break, np.array([340,341])])
                
                rE, rP, rS, rV, rD, r_mem, r_var = np.split(R, ind_break, axis=1)
                
                avg_start = len(r_mem)//2
                M_steady_state[cond_num,seed] = np.mean(r_mem[-avg_start:])
                V_steady_state[cond_num,seed] = np.mean(r_var[-avg_start:])
        
        # save data
        with open(file_data,'wb') as f:
            pickle.dump([num_seeds, SDs, M_steady_state, V_steady_state],f)
            
    else:
        
        # load data
        with open(file_data,'rb') as f:
            [num_seeds, SDs, M_steady_state, V_steady_state] = pickle.load(f)
            
    ### plot
    _, ax = plt.subplots(1,1)
    
    M_avg = np.mean((M_steady_state - 5)/5 * 100, 1)
    V_avg = np.mean((V_steady_state - 4)/4 * 100, 1)
    
    M_sem = np.std((M_steady_state - 5)/5 * 100, 1)/np.sqrt(num_seeds)
    V_sem = np.std((V_steady_state - 4)/4 * 100, 1)/np.sqrt(num_seeds)
    
    ax.plot(SDs, M_avg, 'b')
    ax.fill_between(SDs, M_avg - M_sem/2, M_avg + M_sem/2, color='b', alpha=0.5)
    
    ax.plot(SDs, V_avg, 'r')
    ax.fill_between(SDs, V_avg - V_sem/2, V_avg + V_sem/2, color='r', alpha=0.5)
    
    ax.set_ylim(bottom=0)

    ax.set_xlabel('Uncorrelated deviation (SD)')
    ax.set_ylabel('Deviation (%)')


# %% Run example 

run_cell = False

if run_cell:
    
    # get data
    with open('../results/data/population/data_population_network_parameters.pickle','rb') as f:
        [neurons_visual, inp_ext_soma, inp_ext_dend, weight_name, Dict_w, Dict_t] = pickle.load(f)
    
    with open('../results/data/population/data_population_network_para4corr_gain.pickle','rb') as f:
        [_, _, gain_factors_nPE_5, gain_factors_pPE_5, nPE_true, pPE_true] = pickle.load(f)
        
    # parametrisation neurons, network and rates
    NeuPar = Neurons()
    RatePar = Activity_Zero(NeuPar)
    NetPar = Network(NeuPar, Dict_w, Dict_t, weight_name, neurons_visual, gain_factors_nPE = gain_factors_nPE_5, 
                     gain_factors_pPE = gain_factors_pPE_5, nPE_true = nPE_true, pPE_true = pPE_true)
    
    StimPar = Stimulation(5, 2, inp_ext_soma, inp_ext_dend, neurons_visual, trial_duration = np.int32(200000), 
                          num_values_per_trial = np.int32(400)) # 20000, 20
    
    # run network
    run_population_net(NeuPar, NetPar, StimPar, RatePar, 0.1, folder='population', fln='Example')
    
    
    # load simulation data
    PathData = '../results/data/population'
    arr = np.loadtxt(PathData + '/Data_PopulationNetwork_Example.dat',delimiter=' ')
    t, R = arr[:,0], arr[:, 1:]
    
    ind_break = np.cumsum(NeuPar.NCells,dtype=np.int32)
    ind_break = np.concatenate([ind_break, np.array([340,341])])
    
    rE, rP, rS, rV, rD, r_mem, r_var = np.split(R, ind_break, axis=1)
    
    # plot data
    _, axs = plt.subplots(2,3, tight_layout=True, sharex=True)
    
    axs[0,0].plot(t, rE)
    axs[0,0].set_title('E')
    
    axs[0,1].plot(t, rD)
    axs[0,1].set_title('D')
    
    axs[0,2].plot(t, r_mem, 'b')
    axs[0,2].plot(t, r_var, 'r')
    axs[0,2].set_title('M (blue) & V(red)')
    
    axs[1,0].plot(t, rP)
    axs[1,0].set_title('P')
    
    axs[1,1].plot(t, rS)
    axs[1,1].set_title('S')
    
    axs[1,2].plot(t, rV)
    axs[1,2].set_title('V')
    

# %% Find nPE and pPE neurons

run_cell = False

if run_cell:
    
    # get data
    fln = '../results/data/population/data_population_network_parameters.pickle'
    
    with open(fln,'rb') as f:
        [neurons_visual, inp_ext_soma, inp_ext_dend, weight_name, Dict_w, Dict_t] = pickle.load(f)
    
    # parametrisation neurons, network and rates
    NeuPar = Neurons()
    RatePar = Activity_Zero(NeuPar)
    NetPar = Network(NeuPar, Dict_w, Dict_t, weight_name, neurons_visual)
       
    # take out connections from PE neurons to M or V
    NetPar.wME *= 0
    NetPar.wVarE *= 0
    
    
    ### First S>P
    StimPar = Stimulation(5, 1e-3, inp_ext_soma, inp_ext_dend, neurons_visual, trial_duration = np.int32(10000), 
                          num_values_per_trial = np.int32(20))
    
    RatePar.r_mem0 = np.array([0], dtype=dtype)
    
    # run network
    run_population_net(NeuPar, NetPar, StimPar, RatePar, 0.1, folder='population', fln='test')
    
    
    # load simulation data
    PathData = '../results/data/population'
    arr = np.loadtxt(PathData + '/Data_PopulationNetwork_test.dat',delimiter=' ')
    t_under, R_under = arr[:,0], arr[:, 1:]
    
    
    ### Second S<P
    StimPar = Stimulation(0, 1e-3, inp_ext_soma, inp_ext_dend, neurons_visual, trial_duration = np.int32(10000), 
                          num_values_per_trial = np.int32(20))
    
    RatePar.r_mem0 = np.array([5], dtype=dtype)
    
    # run network
    run_population_net(NeuPar, NetPar, StimPar, RatePar, 0.1, folder='population', fln='test')
    
    
    # load simulation data
    PathData = '../results/data/population'
    arr = np.loadtxt(PathData + '/Data_PopulationNetwork_test.dat',delimiter=' ')
    t_over, R_over = arr[:,0], arr[:, 1:]
    
    
    ### Find nPE and pPE neurons
    ind_break = np.cumsum(NeuPar.NCells,dtype=np.int32)
    ind_break = np.concatenate([ind_break, np.array([340,341])])
    
    rE_under = R_under[:,:NeuPar.NCells[0]]
    rE_over = R_over[:,:NeuPar.NCells[0]]
    
    nPE_true = ((rE_over[-1,:] - rE_under[-1,:]) > 0)
    pPE_true = ((rE_under[-1,:] - rE_over[-1,:]) > 0)

    ### cumulative response strength and individual gain factors
    sum_nPE_response_diff_5 = sum(rE_over[-1,nPE_true])
    sum_pPE_response_diff_5 = sum(rE_under[-1,pPE_true])
    
    gain_factors_nPE_5 = np.ones(NeuPar.NCells[0], dtype=dtype)
    gain_factors_pPE_5 = np.ones(NeuPar.NCells[0], dtype=dtype)
    gain_factors_nPE_5[nPE_true] = (5/rE_over[-1,:])[nPE_true]
    gain_factors_pPE_5[pPE_true] = (5/rE_under[-1,:])[pPE_true]
    
    ### save data
    file_data = '../results/data/population/data_population_network_para4corr_gain.pickle'
    
    with open(file_data,'wb') as f:
        pickle.dump([sum_nPE_response_diff_5, sum_pPE_response_diff_5, gain_factors_nPE_5, gain_factors_pPE_5, nPE_true, pPE_true],f)
        

# %% Test run --> without connections from PE to M or V

run_cell = False

if run_cell:

    # get data
    fln = '../results/data/population/data_population_network_parameters.pickle'
    
    with open(fln,'rb') as f:
        [neurons_visual, inp_ext_soma, inp_ext_dend, weight_name, Dict_w, Dict_t] = pickle.load(f)
    
    # parametrisation neurons, network and rates
    NeuPar = Neurons()
    RatePar = Activity_Zero(NeuPar)
    NetPar = Network(NeuPar, Dict_w, Dict_t, weight_name, neurons_visual)
       
    # take out connections from PE neurons to M or V
    NetPar.wME *= 0
    NetPar.wVarE *= 0
    
    # define stimulation
    #inp_ext_soma[:NeuPar.NCells[0]] = 4.4
    
    StimPar = Stimulation(0, 1e-3, inp_ext_soma, inp_ext_dend, neurons_visual, trial_duration = np.int32(10000), 
                          num_values_per_trial = np.int32(20))
    
    RatePar.r_mem0 = np.array([0], dtype=dtype)
    
    # run network
    run_population_net(NeuPar, NetPar, StimPar, RatePar, 0.1, folder='population', fln='test')
    
    
    # load simulation data
    PathData = '../results/data/population'
    arr = np.loadtxt(PathData + '/Data_PopulationNetwork_test.dat',delimiter=' ')
    t, R = arr[:,0], arr[:, 1:]
    
    ind_break = np.cumsum(NeuPar.NCells,dtype=np.int32)
    ind_break = np.concatenate([ind_break, np.array([340,341])])
    
    rE, rP, rS, rV, rD, r_mem, r_var = np.split(R, ind_break, axis=1)
    
    # plot data
    _, axs = plt.subplots(2,3, tight_layout=True, sharex=True)
    
    axs[0,0].plot(t, rE)
    axs[0,0].set_title('E')
    
    axs[0,1].plot(t, rD)
    axs[0,1].set_title('D')
    
    axs[0,2].plot(t, r_mem, 'b')
    axs[0,2].plot(t, r_var, 'r')
    axs[0,2].set_title('M (blue) & V(red)')
    
    axs[1,0].plot(t, rP)
    axs[1,0].set_title('P')
    
    axs[1,1].plot(t, rS)
    axs[1,1].set_title('S')
    
    axs[1,2].plot(t, rV)
    axs[1,2].set_title('V')
 
# is consistent with other implementation (from previous paper, 2022) --> juhuu
