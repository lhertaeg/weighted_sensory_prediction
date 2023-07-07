#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% import

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path
import seaborn as sns

from matplotlib.collections import LineCollection

from src.plot_data import plot_impact_para

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_3_S2.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(9/inch,3/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

ax_A = fig.add_subplot(G[0,0])
ax_A.text(-0.4, 1.2, 'A', transform=ax_A.transAxes, fontsize=fs+1)
plt.setp(ax_A.get_xticklabels(), visible=False)

ax_B = fig.add_subplot(G[0,1])
ax_B.text(-0.2, 1.2, 'B', transform=ax_B.transAxes, fontsize=fs+1)
plt.setp(ax_B.get_xticklabels(), visible=False)

# ax_B1 = fig.add_subplot(G2[0,0])
# ax_B1.set_title(r'1 $\longrightarrow$ 2', fontsize=fs, pad=10)
# ax_B2 = fig.add_subplot(G2[0,1])
# plt.setp(ax_B2.get_yticklabels(), visible=False)
# ax_B2.set_title(r'2 $\longrightarrow$ 3', fontsize=fs, pad=10)

# %% Load control case and define variabilties tested

variability_within = np.array([0, 0.75, 1.5, 2.25, 3])
variability_across = np.array([3, 2.25, 1.5, 0.75, 0])

file_for_data = '../results/data/weighting/data_weighting_control.pickle'

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data,'rb') as f:
        [stimuli, n_ctrl, gain_w_PE_to_P_ctrl, gain_v_PE_to_P_ctrl, 
         add_input_ctrl, id_cells_modulated_ctrl, weight_ctrl] = pickle.load(f)

# %% Impact of update speed

# data files
file_for_data = '../results/data/weighting/data_weighting_connectivity_lower.pickle'

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data,'rb') as f:
        [gains_lower, weights_mod_con_lower] = pickle.load(f)
        
    # plot data
    label_text = ['ctrl', '1', '2']
    weights_modulated = np.vstack((weights_mod_con_lower[:,0], weights_mod_con_lower[:,-1]))
    plot_impact_para(variability_across, weight_ctrl, weights_modulated, para_range_tested=[gains_lower[0], gains_lower[-1]], 
                      label_text=label_text, ms=3, ax=ax_A)

# %% Impact of activation function

# data files
file_for_data = '../results/data/weighting/data_weighting_activation_function.pickle'

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_for_data,'rb') as f:
            [_, _, _, _, _, _, weight_act] = pickle.load(f)
        
    
    # plot data
    label_text = [r'f(x) = x$^2$', 'f(x) = x']
    plot_impact_para(variability_across, weight_ctrl, weight_act, plot_ylabel=False, ms=3, label_text=label_text, ax=ax_B)

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
