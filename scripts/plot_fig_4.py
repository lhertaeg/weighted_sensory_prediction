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

from src.plot_data import plot_neuromod_impact_inter, illustrate_sensory_weight_variance, plot_changes_upon_input2PE_neurons
from src.plot_data import plot_illustration_changes_upon_baseline_PE, plot_illustration_changes_upon_gain_PE, plot_influence_interneurons_gain_baseline
from src.plot_data import plot_legend_illustrations

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_4.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(18/inch,18/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(2, 1, figure=fig, hspace=0.6, height_ratios=[1.2,2])

G1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G[0,0], width_ratios=[3,1], hspace=0.5)
G2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[1,0], width_ratios=[2.5,1], wspace=0.4)

G11 = gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec=G1[:,0])#, height_ratios=[1,3], hspace=0.7)

ax_A = fig.add_subplot(G11[:,:])
ax_A.axis('off')
ax_A.set_title('Neuromodulators acting on interneurons can bias the weighting of sensory inputs and predictions', fontsize=fs, pad=10)
ax_A.text(-0.15, 1.15, 'A', transform=ax_A.transAxes, fontsize=fs+1)

ax_A12 = fig.add_subplot(G11[0,0])
plt.setp(ax_A12.get_xticklabels(), visible=False)
ax_A13 = fig.add_subplot(G11[0,1])
plt.setp(ax_A13.get_yticklabels(), visible=False)
plt.setp(ax_A13.get_xticklabels(), visible=False)
ax_A14 = fig.add_subplot(G11[0,2])
plt.setp(ax_A14.get_yticklabels(), visible=False)
plt.setp(ax_A14.get_xticklabels(), visible=False)

ax_A22 = fig.add_subplot(G11[1,0])
ax_A23 = fig.add_subplot(G11[1,1])
plt.setp(ax_A23.get_yticklabels(), visible=False)
ax_A24 = fig.add_subplot(G11[1,2])
plt.setp(ax_A24.get_yticklabels(), visible=False)

ax_A_legend = fig.add_subplot(G1[0,1])
ax_A_legend.axis('off')

ax_B = fig.add_subplot(G1[1,1])
ax_B.text(-0.6, 1.15, 'B', transform=ax_B.transAxes, fontsize=fs+1)
#ax_B.set_title('Sensory weight as a function \nof stimulus and trial variance', fontsize=fs, pad=10)

G21 = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=G2[0,1], height_ratios=[1,3], hspace=0.7)

ax_F = fig.add_subplot(G21[1,0])
ax_F.set_title('Effect of interneuron activation \non baseline and gain of PE neurons', fontsize=fs, pad=10)
ax_F.text(-0.4, 1.15, 'E', transform=ax_F.transAxes, fontsize=fs+1)

G22 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=G2[0,0], width_ratios=[1,2], wspace=0.6)#, wspace=0.3, hspace=0.5)
G221 = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=G22[0,0])
G222 = gridspec.GridSpecFromSubplotSpec(3,2, subplot_spec=G22[0,1])

ax_C = fig.add_subplot(G221[:,0])
ax_C.axis('off')
ax_C.set_title('Perturbing PE neurons', fontsize=fs, pad=10)
ax_C.text(-0.5, 1.15, 'C', transform=ax_C.transAxes, fontsize=fs+1)

ax_C1 = fig.add_subplot(G221[0,0])
plt.setp(ax_C1.get_xticklabels(), visible=False)
ax_C2 = fig.add_subplot(G221[1,0])
plt.setp(ax_C2.get_xticklabels(), visible=False)
ax_C3 = fig.add_subplot(G221[2,0])

ax_D = fig.add_subplot(G222[:,:])
ax_D.axis('off')
ax_D.set_title('Illustration of changes in mean and variance \nwith changes in baseline (left) and gain (right)', fontsize=fs, pad=10)
ax_D.text(-0.2, 1.15, 'D', transform=ax_D.transAxes, fontsize=fs+1)

ax_D1 = fig.add_subplot(G222[0,0])
ax_D2 = fig.add_subplot(G222[1,0])
ax_D3 = fig.add_subplot(G222[2,0])

ax_E1 = fig.add_subplot(G222[0,1])
ax_E2 = fig.add_subplot(G222[1,1])
ax_E3 = fig.add_subplot(G222[2,1])

ax_E_legend = fig.add_subplot(G21[0,0])
ax_E_legend.axis('off')

# %% Neuromodulators acting on interneurons for 2 limit cases
    
plot_neuromod_impact_inter(0, 1, 0, s=7, ax1=ax_A12, ax2=ax_A13, ax3=ax_A14, flg_plot_xlabel=False, flg_plot_bars=False)
plot_neuromod_impact_inter(0, 0, 1, s=7, ax1=ax_A22, ax2=ax_A23, ax3=ax_A24, flg_plot_xlabel=False)

# plot legend
markers = ['o', 's', 'd']
label_text = [r'MFN 1 (SOM$\leftarrow$ S, VIP$\leftarrow$ P)', 
              r'MFN 2 (SOM$\leftarrow$ P, VIP$\leftarrow$ S)', 
              r'MFN 3 (SOM$\leftarrow$ S, VIP$\leftarrow$ S)']

for i in range(3):
    ax_A_legend.plot(np.nan,np.nan, marker=markers[i], color='k', label=label_text[i], ms=2.5, ls='None')
    
ax_A_legend.legend(loc=2, fontsize=fs, frameon=False, bbox_to_anchor=(-0.3, 0.9))

# %% Sensory weight as a function of variance

illustrate_sensory_weight_variance(ax=ax_B)

# %% How are the v & m neurons influenced by changes in nPE and pPE neurons in the lower and higher PE circuit?

plot_changes_upon_input2PE_neurons(ax1 = ax_C1, ax2 = ax_C2, ax3 = ax_C3)

# %%  How is the variance influenced by BL of nPE and pPE in lower and higher PE circuit?    
 
plot_illustration_changes_upon_baseline_PE(BL = np.linspace(0,3,7), ax1 = ax_D1, ax2 = ax_D2, ax3 = ax_D3)

# %%  How is the variance influenced by gain of nPE and pPE in lower and higher PE circuit?    
 
plot_illustration_changes_upon_gain_PE(gains=np.linspace(0.5,1.5,7), ax1 = ax_E1, ax2 = ax_E2, ax3 = ax_E3)
plot_legend_illustrations(ax_E_legend)


# %% How do the interneurons influence the PE neuron properties?

plot_influence_interneurons_gain_baseline(ax=ax_F)

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
