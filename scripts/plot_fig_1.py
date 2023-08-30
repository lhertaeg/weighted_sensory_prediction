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

from src.plot_data import plot_nPE_pPE_activity_compare

# %% Universal parameters

fs = 7
inch = 2.54

# %% Define files and paths

figure_name = 'Fig_1_partial.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)
    

# %% Define figure structure

figsize=(5.5/inch,2/inch)
fig = plt.figure(figsize=figsize)

A = gridspec.GridSpec(1, 2, figure=fig, wspace=0.6)

# ax_A =  fig.add_subplot(A[:,:])
# ax_A.axis('off')
# ax_A.set_xlabel('Sens - Pred (1/s)', fontsize=fs)

ax_1 = fig.add_subplot(A[0,0])
ax_2 = fig.add_subplot(A[0,1])


# %% Show Activity of nPE and pPE, compare with (S-P)**2

# data files
file_for_data = '../results/data/moments/data_mfn_10_PE_neurons_constant_stimuli_P_fixed.pickle'

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data,'rb') as f:
        [prediction_initial, stimulus_tested, nPE, pPE] = pickle.load(f)
    
    # plot data
    plot_nPE_pPE_activity_compare(prediction_initial, stimulus_tested, nPE, pPE, ax1 = ax_1, ax2 = ax_2, ms=3, lw=2, fs=fs)

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
