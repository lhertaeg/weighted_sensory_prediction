#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

from src.functions_simulate import simulate_neuromod
from src.plot_data import plot_heatmap_neuromod, plot_combination_activation_INs

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32


# %% import

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path
import seaborn as sns

from matplotlib.collections import LineCollection

from src.plot_data import plot_neuromod_impact_inter, illustrate_sensory_weight_variance, plot_changes_upon_input2PE_neurons, plot_illustration_neuromod_results
from src.plot_data import plot_illustration_changes_upon_baseline_PE, plot_illustration_changes_upon_gain_PE, plot_influence_interneurons_gain_baseline
from src.plot_data import plot_legend_illustrations, plot_changes_upon_input2PE_neurons_new, plot_influence_interneurons_baseline_or_gain, plot_influence_interneurons_baseline_or_gain
from src.plot_data import plot_standalone_colorbar

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_4.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(18/inch,14/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(2, 1, figure=fig, hspace=0.7) #, hspace=0.6, height_ratios=[1.2,2])
A = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=G[0,0])#, width_ratios=[3,1], hspace=0.5)

G_sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[1,0], width_ratios=[0.7,1], wspace=0.3)
C_and_D = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G_sub[0,1], wspace=0.7, hspace=0.5)

ax_A = fig.add_subplot(A[:,:4])
ax_A.axis('off')
ax_A.set_title('Neuromodulators acting on interneurons can bias the weighting of sensory inputs and predictions', fontsize=fs, pad=10)
ax_A.text(-0.1, 1.15, 'A', transform=ax_A.transAxes, fontsize=fs+1)

ax_A12 = fig.add_subplot(A[0,0])
plt.setp(ax_A12.get_xticklabels(), visible=False)
ax_A13 = fig.add_subplot(A[0,1])
plt.setp(ax_A13.get_yticklabels(), visible=False)
plt.setp(ax_A13.get_xticklabels(), visible=False)
ax_A14 = fig.add_subplot(A[0,2])
plt.setp(ax_A14.get_yticklabels(), visible=False)
plt.setp(ax_A14.get_xticklabels(), visible=False)

ax_A22 = fig.add_subplot(A[1,0])
ax_A23 = fig.add_subplot(A[1,1])
plt.setp(ax_A23.get_yticklabels(), visible=False)
ax_A24 = fig.add_subplot(A[1,2])
plt.setp(ax_A24.get_yticklabels(), visible=False)

ax_A_legend = fig.add_subplot(A[1,3])
ax_A_legend.axis('off')

ax_B = fig.add_subplot(G_sub[0,0])
ax_B.text(-0.3, 1.1, 'B', transform=ax_B.transAxes, fontsize=fs+1)

ax_CD = fig.add_subplot(C_and_D[:,:])
ax_CD.axis('off')
ax_CD.set_title(r'Sensory weight $\longleftarrow$ variance neuron $\longleftarrow$ PE neurons $\longleftarrow$ interneurons', fontsize=fs, pad=20)

ax_C1 = fig.add_subplot(C_and_D[0,0])
ax_C1.text(-0.45, 1.15, 'C', transform=ax_C1.transAxes, fontsize=fs+1)
ax_D1 = fig.add_subplot(C_and_D[0,1])
ax_D1.text(-0.55, 1.15, 'D', transform=ax_D1.transAxes, fontsize=fs+1)
ax_C2 = fig.add_subplot(C_and_D[1,0])
plt.setp(ax_C1.get_xticklabels(), visible=False)
ax_D2 = fig.add_subplot(C_and_D[1,1])

# %% Neuromodulators acting on interneurons for 2 limit cases
    
plot_neuromod_impact_inter(0, 1, 0, s=7, ax1=ax_A12, ax2=ax_A13, ax3=ax_A14, flg_plot_xlabel=False, flg_plot_bars=False)
plot_neuromod_impact_inter(0, 0, 1, s=7, ax1=ax_A22, ax2=ax_A23, ax3=ax_A24, flg_plot_xlabel=False)

# plot legend
markers = ['o', 's', 'd']
label_text = [r'#1 (SOM$\leftarrow$ S, VIP$\leftarrow$ P)', 
              r'#2 (SOM$\leftarrow$ P, VIP$\leftarrow$ S)', 
              r'#3 (SOM$\leftarrow$ S, VIP$\leftarrow$ S)']

for i in range(3):
    ax_A_legend.plot(np.nan,np.nan, marker=markers[i], color='k', label=label_text[i], ms=2.5, ls='None')
    
ax_A_legend.legend(loc=2, fontsize=fs, frameon=False, bbox_to_anchor=(-0.1, 1), 
                   title='Mean-field networks tested', title_fontsize=fs)


# plot colorbar
plot_standalone_colorbar(ax_A_legend)
 

# %% Illustrate main results

plot_illustration_neuromod_results(ax=ax_B)


# %% How are the vv neurons influenced by changes in nPE and pPE neurons in the lower and higher PE circuit?

plot_changes_upon_input2PE_neurons_new(ax1 = ax_C1, ax2 = ax_C2)


# %% How do the interneurons influence the PE neuron properties?

plot_influence_interneurons_baseline_or_gain(ax=ax_D1, plot_annotation=False)
plot_influence_interneurons_baseline_or_gain(plot_baseline=False, plot_annotation=False, ax=ax_D2)

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
