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

from src.plot_data import plot_M_and_V_for_population_example, plot_deviation_in_population_net

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_2_S2.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)
    
    
# %% Define figure structure

figsize=(15/inch,10/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(2, 1, figure=fig, hspace=1)
G1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[0,0], width_ratios=[2,1], wspace=0.5)
G2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=G[1,0], wspace=0.4)

ax_A = fig.add_subplot(G1[0,0])
ax_A.axis('off')
ax_A.text(-0.1, 1.2, 'A', transform=ax_A.transAxes, fontsize=fs+1)

ax_B = fig.add_subplot(G1[0,1])
ax_B.set_title('M and V neuron activity for a population network', fontsize=fs, pad=10)
ax_B.text(-0.6, 1.2, 'B', transform=ax_B.transAxes, fontsize=fs+1)

ax_D = fig.add_subplot(G2[0,1])
plt.setp(ax_D.get_yticklabels(), visible=False)
ax_D.set_title('Deviations for V neuron are \ndependent of correlated changes', fontsize=fs, pad=10)
ax_D.text(-0.2, 1.3, 'D', transform=ax_D.transAxes, fontsize=fs+1)

ax_C = fig.add_subplot(G2[0,0], sharey=ax_D)
ax_C.set_title('Deviations are independent \nof uncorrelated changes', fontsize=fs, pad=10)
ax_C.text(-0.2, 1.3, 'C', transform=ax_C.transAxes, fontsize=fs+1)

ax_E = fig.add_subplot(G2[0,2], sharey=ax_D)
plt.setp(ax_E.get_yticklabels(), visible=False)
ax_E.set_title('Deviations are largely \nindependent of network sparsity', fontsize=fs, pad=10)
ax_E.text(-0.2, 1.3, 'E', transform=ax_E.transAxes, fontsize=fs+1)


# %% Show example 

# data files
file_for_data = '../results/data/population/Data_PopulationNetwork_Example.dat'

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    NCells = [140, 20, 20, 20]
    arr = np.loadtxt(file_for_data,delimiter=' ')
    t, R = arr[:,0], arr[:, 1:]

    ind_break = np.cumsum(NCells,dtype=np.int32)
    ind_break = np.concatenate([ind_break, np.array([340,341])])

    rE, rP, rS, rV, rD, r_mem, r_var = np.split(R, ind_break, axis=1)
    
    plot_M_and_V_for_population_example(t, r_mem, r_var, ax=ax_B)


# %% Uncorrelated changes

# data files
file_data = '../results/data/population/data_population_uncorrelated_deviations.pickle'

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_data,'rb') as f:
        [num_seeds, SDs, M_steady_state, V_steady_state] = pickle.load(f)

    plot_deviation_in_population_net(SDs, num_seeds, M_steady_state, V_steady_state, r'Uncorrelated dev, SD of $\gamma$', ax=ax_C, ylim=None)
    

# %% Correlated changes

# data files
file_data = '../results/data/population/data_population_correlated_deviations.pickle'

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_data,'rb') as f:
        [num_seeds, means, M_steady_state, V_steady_state] = pickle.load(f)

    plot_deviation_in_population_net(means, num_seeds, M_steady_state, V_steady_state, r'Correlated dev, mean of $\gamma$', ax=ax_D, 
                                     plt_ylabel=False, ylim=None)
    

# %% Sparsity

# data files
file_data = '../results/data/population/data_population_sparsity.pickle'

if not os.path.exists(file_for_data):
    print('Data does not exist yet. Please run corresponding file.')
else:
    
    # load data
    with open(file_data,'rb') as f:
        [num_seeds, p_conns, M_steady_state, V_steady_state] = pickle.load(f)

    plot_deviation_in_population_net(p_conns, num_seeds, M_steady_state, V_steady_state, 'Sparsity (connection prop.)', ax=ax_E, 
                                     plt_ylabel=False, ylim=None)

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
    