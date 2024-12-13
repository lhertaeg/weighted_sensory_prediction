#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:10:23 2024

@author: loreen.hertaeg
"""

# %% import

import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path
import numpy as np

from src.plot_data import plot_output_different_weightings


# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_S10.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(12/inch,10/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(3, 1, figure=fig, wspace=0.1)
A = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G[0,0])
B = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G[1,0])
C = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G[2,0])

ax_A1 = fig.add_subplot(A[0,0])
ax_A1.text(-0.4, 1., 'A', transform=ax_A1.transAxes, fontsize=fs+1)
ax_A2 = fig.add_subplot(A[0,1])
ax_A3 = fig.add_subplot(A[0,2])

ax_B1 = fig.add_subplot(B[0,0], sharey=ax_A1)
ax_B1.text(-0.4, 1., 'B', transform=ax_B1.transAxes, fontsize=fs+1)
ax_B2 = fig.add_subplot(B[0,1], sharey=ax_A2, sharex=ax_A2)
ax_B3 = fig.add_subplot(B[0,2], sharey=ax_A3, sharex=ax_A3)

ax_C1 = fig.add_subplot(C[0,0], sharey=ax_A1)
ax_C1.text(-0.4, 1., 'C', transform=ax_C1.transAxes, fontsize=fs+1)
ax_C2 = fig.add_subplot(C[0,1], sharey=ax_A2, sharex=ax_A2)
ax_C3 = fig.add_subplot(C[0,2], sharey=ax_A3, sharex=ax_A3)

plt.setp(ax_A1.get_xticklabels(), visible=False)
plt.setp(ax_A2.get_xticklabels(), visible=False)
plt.setp(ax_A3.get_xticklabels(), visible=False)
plt.setp(ax_B1.get_xticklabels(), visible=False)
plt.setp(ax_B2.get_xticklabels(), visible=False)
plt.setp(ax_B3.get_xticklabels(), visible=False)

axs = np.array([[ax_A1, ax_A2, ax_A3], [ax_B1, ax_B2, ax_B3], [ax_C1, ax_C2, ax_C3]])

# %%  Run example in which the network faces a sudden transition to a state with high sensory noise 
# compare different approaches

file_for_data = '../results/data/weighting/data_transition_10_01000_5503_compare.pickle'

with open(file_for_data,'rb') as f:
    [n_trials, trial_duration, _, stimuli, m_neuron_lower, v_neuron_lower, 
     m_neuron_higher, v_neuron_higher, alpha, beta, weighted_output] = pickle.load(f)

plot_output_different_weightings(n_trials, trial_duration, stimuli, m_neuron_lower, v_neuron_lower, m_neuron_higher, 
                                 v_neuron_higher, weighted_output, axs=axs)


# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
