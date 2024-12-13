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
import matplotlib.gridspec as gridspec
import os.path

from src.plot_data import plot_robustness

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_S6.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)
    
    
# %% Define figure structure

figsize=(18/inch,15/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 6, 3], wspace=0.5)
Col1 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=G[0,0], hspace=0.3)
Col2 = gridspec.GridSpecFromSubplotSpec(5, 2, subplot_spec=G[0,1], wspace=0.3, hspace=0.3)
Col3 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=G[0,2], hspace=0.3)

#ax_A1 = fig.add_subplot(Col1[0,0])
ax_A2 = fig.add_subplot(Col2[0,0])
ax_A3 = fig.add_subplot(Col2[0,1])
ax_A4 = fig.add_subplot(Col3[0,0])
ax_A = np.array([ax_A2, ax_A3, ax_A4])

#ax_B1 = fig.add_subplot(Col1[1,0])
ax_B2 = fig.add_subplot(Col2[1,0])
ax_B3 = fig.add_subplot(Col2[1,1])
ax_B4 = fig.add_subplot(Col3[1,0])
ax_B = np.array([ax_B2, ax_B3, ax_B4])

#ax_C1 = fig.add_subplot(Col1[2,0])
ax_C2 = fig.add_subplot(Col2[2,0])
ax_C3 = fig.add_subplot(Col2[2,1])
ax_C4 = fig.add_subplot(Col3[2,0])
ax_C = np.array([ax_C2, ax_C3, ax_C4])

#ax_D1 = fig.add_subplot(Col1[3,0])
ax_D2 = fig.add_subplot(Col2[3,0])
ax_D3 = fig.add_subplot(Col2[3,1])
ax_D4 = fig.add_subplot(Col3[3,0])
ax_D = np.array([ax_D2, ax_D3, ax_D4])

#ax_E1 = fig.add_subplot(Col1[4,0])
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


# %% Robustness

folder = '../results/data/moments/'

## tau_V
file_for_data = folder + 'data_change_tauV.pickle'
with open(file_for_data,'rb') as f:
        [paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
         m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
        
plot_robustness(m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l, alpha_s, alpha_l, xlabel1 = ' ', xlabel2 = ' ', 
                title1 = 'Lower-level M neuron', title2 = 'Lower-level V neuron', title3 = 'Sensory weight of full network', ylabel1=' ', ylabel2=' ', axs=ax_A)   

## wPE2P
file_for_data = folder + 'data_change_wPE2P.pickle'
with open(file_for_data,'rb') as f:
        [paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
         m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
        
plot_robustness(m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l, alpha_s, alpha_l, xlabel1 = ' ', xlabel2 = ' ', 
                ylabel1=' ', ylabel2=' ', axs=ax_B) 

## wP2PE
file_for_data = folder + 'data_change_wP2PE.pickle'
with open(file_for_data,'rb') as f:
        [paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
         m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
        
plot_robustness(m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l, alpha_s, alpha_l, xlabel1 = ' ', xlabel2 = ' ', axs=ax_C)     

## wPE2V
file_for_data = folder + 'data_change_wPE2V.pickle'
with open(file_for_data,'rb') as f:
        [paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
         m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
        
plot_robustness(m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l, alpha_s, alpha_l, xlabel1 = ' ', xlabel2 = ' ', 
                ylabel1=' ', ylabel2=' ', axs=ax_D)

## top-down
file_for_data = folder + 'data_add_top_down.pickle'
with open(file_for_data,'rb') as f:
        [paras, alpha_s, alpha_l, m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l] = pickle.load(f)
        
plot_robustness(m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l, alpha_s, alpha_l, ylabel1=' ', ylabel2=' ', axs=ax_E)  

    
# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
    