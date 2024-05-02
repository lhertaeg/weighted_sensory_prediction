#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:58:33 2024

@author: loreen.hertaeg
"""

# %% Import

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path

from src.plot_data import illustrate_PE_establish_M


# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_1_S2.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)
    

# %% Define figure structure
    
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
    
# %% Illustration how nPE and pPE neurons establish M

file_for_data = '../results/data/moments/data_PE_establish_M_10.pickle'

with open(file_for_data,'rb') as f:
    [_, _, trial_duration, num_values_per_trial, stimuli, m_neuron, v_neuron, PE] = pickle.load(f)
    

illustrate_PE_establish_M(m_neuron, PE, stimuli, trial_duration, num_values_per_trial, [0, 20000], [180000, 200000], axs=axs)
    
    
# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)