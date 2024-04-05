#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:34:53 2024

@author: loreen.hertaeg
"""

# %% import

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path

from src.plot_data import illustrate_PE_establish_M, plot_impact_baseline, plot_example_mean, plot_example_variance 
from src.plot_data import plot_dev_cntns, plot_example_stimuli_smoothed, plot_robustness

fs = 6
inch = 2.54

# %% Illustration how nPE and pPE neurons establish M

run_cell = False

if run_cell:

    file_for_data = '../results/data/moments/data_PE_establish_M_10.pickle'
    
    with open(file_for_data,'rb') as f:
        [_, _, trial_duration, num_values_per_trial, stimuli, m_neuron, v_neuron, PE] = pickle.load(f)
    
    
    figsize=(12/inch,10/inch)
    fig = plt.figure(figsize=figsize)
    
    G = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.2)
    
    ax_A1 = fig.add_subplot(G[0,0])
    ax_A2 = fig.add_subplot(G[0,1], sharey=ax_A1)
    ax_A1.text(-0.25, 1.1, 'A', transform=ax_A1.transAxes, fontsize=fs+1)
    plt.setp(ax_A2.get_yticklabels(), visible=False)
    
    ax_B1 = fig.add_subplot(G[1,0])
    ax_B2 = fig.add_subplot(G[1,1], sharey=ax_B1)
    ax_B1.text(-0.25, 1.1, 'B', transform=ax_B1.transAxes, fontsize=fs+1)
    plt.setp(ax_B2.get_yticklabels(), visible=False)
    
    ax_C1 = fig.add_subplot(G[2,0], sharey=ax_B1)
    ax_C2 = fig.add_subplot(G[2,1], sharey=ax_B1)
    ax_C1.text(-0.25, 1.1, 'C', transform=ax_C1.transAxes, fontsize=fs+1)
    plt.setp(ax_C2.get_yticklabels(), visible=False)
    
    plt.setp(ax_A1.get_xticklabels(), visible=False)
    plt.setp(ax_B1.get_xticklabels(), visible=False)
    plt.setp(ax_A2.get_xticklabels(), visible=False)
    plt.setp(ax_B2.get_xticklabels(), visible=False)
    
    axs = np.array([[ax_A1, ax_A2], [ax_B1, ax_B2], [ax_C1, ax_C2]])
    
    illustrate_PE_establish_M(m_neuron, PE, stimuli, trial_duration, num_values_per_trial, [0, 20000], [180000, 200000], axs=axs)
    
    
# %% Effect of PE neuron baseline

run_cell = False

if run_cell:

    figsize=(12/inch,9/inch)
    fig = plt.figure(figsize=figsize)
    
    G = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2,1], hspace=0.3)
    AB = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=G[0,0])
    C = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G[1,0])
                          
    ax_A1 = fig.add_subplot(AB[0,0])
    ax_A2 = fig.add_subplot(AB[0,1], sharey=ax_A1)
    ax_A3 = fig.add_subplot(AB[0,2], sharey=ax_A1)
    ax_A1.text(-0.4, 1., 'A', transform=ax_A1.transAxes, fontsize=fs+1)
    plt.setp(ax_A2.get_yticklabels(), visible=False)
    plt.setp(ax_A3.get_yticklabels(), visible=False)
    
    ax_B1 = fig.add_subplot(AB[1,0])
    ax_B2 = fig.add_subplot(AB[1,1], sharey=ax_B1)
    ax_B3 = fig.add_subplot(AB[1,2], sharey=ax_B1) 
    ax_B1.text(-0.4, 1., 'B', transform=ax_B1.transAxes, fontsize=fs+1)
    plt.setp(ax_B2.get_yticklabels(), visible=False)
    plt.setp(ax_B3.get_yticklabels(), visible=False)
    
    ax_C1 = fig.add_subplot(C[0,0])
    ax_C2 = fig.add_subplot(C[0,1], sharey=ax_C1)
    ax_C3 = fig.add_subplot(C[0,2], sharey=ax_C1)
    ax_C1.text(-0.4, 1., 'C', transform=ax_C1.transAxes, fontsize=fs+1)
    plt.setp(ax_C2.get_yticklabels(), visible=False)
    plt.setp(ax_C3.get_yticklabels(), visible=False)
    
    plt.setp(ax_A1.get_xticklabels(), visible=False)
    plt.setp(ax_B1.get_xticklabels(), visible=False)
    plt.setp(ax_A2.get_xticklabels(), visible=False)
    plt.setp(ax_B2.get_xticklabels(), visible=False)
    plt.setp(ax_A3.get_xticklabels(), visible=False)
    plt.setp(ax_B3.get_xticklabels(), visible=False)
    
    axs = np.array([[ax_A1, ax_A2, ax_A3], [ax_B1, ax_B2, ax_B3], [ax_C1, ax_C2, ax_C3]])    

    input_to_increase_baseline = np.linspace(0,0.5,5)
    plot_impact_baseline(input_to_increase_baseline, axs=axs)  
    
    
# %% Continuous signals

run_cell = False

if run_cell:
    
    file_for_example = '../results/data/moments/data_net_example_cntns_input_one_column.pickle'
    file_for_data = '../results/data/moments/data_cntns_input_one_column.pickle'
    
    with open(file_for_example,'rb') as f:
            [stimuli, trial_duration, m_neuron, v_neuron] = pickle.load(f)
            
    with open(file_for_data,'rb') as f:
        [hann_windows, dev_mean, dev_variance] = pickle.load(f)
        
    
    figsize=(13/inch,7/inch)
    fig = plt.figure(figsize=figsize)
    G = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)
    
    ax_A = fig.add_subplot(G[0,0])
    ax_B = fig.add_subplot(G[0,1])
    ax_C = fig.add_subplot(G[1,0])
    ax_D = fig.add_subplot(G[1,1])

    plot_example_stimuli_smoothed([hann_windows[0], hann_windows[-1]], ax=ax_A)

    plot_dev_cntns(hann_windows, dev_mean, dev_variance, ax = ax_B)

    plot_example_mean(stimuli, trial_duration, m_neuron, mse_flg=False, ax1=ax_C)
    plot_example_variance(stimuli, trial_duration, v_neuron, mse_flg=False, ax1=ax_D)
    
    ax_C.set_title(' ')
    ax_D.set_title(' ')
    ax_D.set_ylabel(' ')
    
# %% Robustness

run_cell = True

if run_cell:
    
    figsize=(18/inch,15/inch)
    fig = plt.figure(figsize=figsize)
    
    G = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 6, 3], wspace=0.5)
    Col1 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=G[0,0], hspace=0.3)
    Col2 = gridspec.GridSpecFromSubplotSpec(5, 2, subplot_spec=G[0,1], wspace=0.3, hspace=0.3)
    Col3 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=G[0,2], hspace=0.3)
    
    ax_A1 = fig.add_subplot(Col1[0,0])
    ax_A2 = fig.add_subplot(Col2[0,0])
    ax_A3 = fig.add_subplot(Col2[0,1])
    ax_A4 = fig.add_subplot(Col3[0,0])
    ax_A = np.array([ax_A2, ax_A3, ax_A4])
    
    ax_B1 = fig.add_subplot(Col1[1,0])
    ax_B2 = fig.add_subplot(Col2[1,0])
    ax_B3 = fig.add_subplot(Col2[1,1])
    ax_B4 = fig.add_subplot(Col3[1,0])
    ax_B = np.array([ax_B2, ax_B3, ax_B4])
    
    ax_C1 = fig.add_subplot(Col1[2,0])
    ax_C2 = fig.add_subplot(Col2[2,0])
    ax_C3 = fig.add_subplot(Col2[2,1])
    ax_C4 = fig.add_subplot(Col3[2,0])
    ax_C = np.array([ax_C2, ax_C3, ax_C4])
    
    ax_D1 = fig.add_subplot(Col1[3,0])
    ax_D2 = fig.add_subplot(Col2[3,0])
    ax_D3 = fig.add_subplot(Col2[3,1])
    ax_D4 = fig.add_subplot(Col3[3,0])
    ax_D = np.array([ax_D2, ax_D3, ax_D4])
    
    ax_E1 = fig.add_subplot(Col1[4,0])
    ax_E2 = fig.add_subplot(Col2[4,0])
    ax_E3 = fig.add_subplot(Col2[4,1])
    ax_E4 = fig.add_subplot(Col3[4,0])
    ax_E = np.array([ax_E2, ax_E3, ax_E4])
    
    plt.setp(ax_A2.get_xticklabels(), visible=False)
    plt.setp(ax_A3.get_xticklabels(), visible=False)
    plt.setp(ax_B2.get_xticklabels(), visible=False)
    plt.setp(ax_B3.get_xticklabels(), visible=False)
    plt.setp(ax_C2.get_xticklabels(), visible=False)
    plt.setp(ax_C3.get_xticklabels(), visible=False)
    plt.setp(ax_D2.get_xticklabels(), visible=False)
    plt.setp(ax_D3.get_xticklabels(), visible=False)
    
    ## tau_V
    file_for_data = '../results/data/moments/data_change_tauV.pickle'
    with open(file_for_data,'rb') as f:
            [paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
             m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
            
    plot_robustness(m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l, alpha_s, alpha_l, xlabel1 = ' ', xlabel2 = ' ', 
                    title1 = 'M neuron', title2 = 'V neuron', axs=ax_A)   

    ## wPE2P
    file_for_data = '../results/data/moments/data_change_wPE2P.pickle'
    with open(file_for_data,'rb') as f:
            [paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
             m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
            
    plot_robustness(m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l, alpha_s, alpha_l, xlabel1 = ' ', xlabel2 = ' ', axs=ax_B) 

    ## wP2PE
    file_for_data = '../results/data/moments/data_change_wP2PE.pickle'
    with open(file_for_data,'rb') as f:
            [paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
             m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
            
    plot_robustness(m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l, alpha_s, alpha_l, xlabel1 = ' ', xlabel2 = ' ', axs=ax_C)     
    
    ## wPE2V
    file_for_data = '../results/data/moments/data_change_wPE2V.pickle'
    with open(file_for_data,'rb') as f:
            [paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
             m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
            
    plot_robustness(m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l, alpha_s, alpha_l, xlabel1 = ' ', xlabel2 = ' ', axs=ax_D)
    
    ## top-down
    file_for_data = '../results/data/moments/data_add_top_down.pickle'
    with open(file_for_data,'rb') as f:
            [paras, alpha_s, alpha_l, m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
            
    plot_robustness(m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l, alpha_s, alpha_l, axs=ax_E)  
            

