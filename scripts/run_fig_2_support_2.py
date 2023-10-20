#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from src.default_parameters import Neurons, Activity_Zero, Network, Stimulation
from src.functions_save import load_network_para
from src.functions_networks import run_population_net

from src.plot_data import plot_gain_factors, plot_deviation_in_population_net, plot_M_and_V_for_population_example

dtype = np.float32


# %% Find nPE and pPE neurons and plot gain factors

run_cell = False
plot_only = True

if run_cell:
    
    if not plot_only: # simulate respective network
    
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
        StimPar = Stimulation(5, 1e-3, inp_ext_soma, inp_ext_dend, neurons_visual, num_values_per_trial = np.int32(20))
        
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
            
    else:
        
        ### load data
        with open('../results/data/population/data_population_network_para4corr_gain.pickle','rb') as f:
            [_, _, gain_factors_nPE_5, gain_factors_pPE_5, nPE_true, pPE_true] = pickle.load(f)
            
    
    ### plot
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

# continue here

run_cell = True
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
        p_conns = np.linspace(0.7, 1, 7)
        num_seeds = 10
        
        # initialise arrays
        M_steady_state = np.zeros((len(p_conns), num_seeds))
        V_steady_state = np.zeros((len(p_conns), num_seeds))
        
        for cond_num, p_conn in enumerate(p_conns):
            for seed in range(num_seeds):
                
                print('p_conn = ', p_conn, ', seed = ', seed)
                
                # parametrisation neurons, network and rates
                NeuPar = Neurons()
                RatePar = Activity_Zero(NeuPar)
                
                NetPar = Network(NeuPar, Dict_w, Dict_t, weight_name, neurons_visual, gain_factors_nPE = gain_factors_nPE_5, 
                                 gain_factors_pPE = gain_factors_pPE_5, nPE_true = nPE_true, pPE_true = pPE_true, p_conn=p_conn)
            
                StimPar = Stimulation(5, 2, inp_ext_soma, inp_ext_dend, neurons_visual, seed=seed)
                            
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
    plot_deviation_in_population_net(p_conns, num_seeds, M_steady_state, V_steady_state, 'Sparsity (connection prop.)', plt_ylabel=False, ylim=None)


# %% Correlated deviations  

run_cell = False
plot_only = True

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
                
                StimPar = Stimulation(5, 2, inp_ext_soma, inp_ext_dend, neurons_visual, seed=seed)
                            
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
    plot_deviation_in_population_net(means, num_seeds, M_steady_state, V_steady_state, 'Correlated deviations (mean)', plt_ylabel=False, ylim=None)
    
    
# %% Uncorrelated deviations  

run_cell = False
plot_only = True

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
                
                StimPar = Stimulation(5, 2, inp_ext_soma, inp_ext_dend, neurons_visual, seed=seed) 
                            
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
    plot_deviation_in_population_net(SDs, num_seeds, M_steady_state, V_steady_state, 'Uncorrelated deviations (SD)', ylim=None)


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
    
    StimPar = Stimulation(5, 2, inp_ext_soma, inp_ext_dend, neurons_visual)
    
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
    plot_M_and_V_for_population_example(t, r_mem, r_var)
