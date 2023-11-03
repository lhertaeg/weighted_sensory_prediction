#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% import

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os.path

from src.plot_data import plot_neuromod_impact, plot_standalone_colorbar
from src.plot_data import plot_changes_upon_input2PE_neurons_new, plot_influence_interneurons_baseline_or_gain


# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_4.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(14/inch,14/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5, height_ratios=[1,1.2])
G1 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=G[0,0], width_ratios=[1,1,1,1.2], wspace=0.8)
G2 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=G[1,0], width_ratios=[1,1,1,1.2], wspace=0.8, hspace=0.7)

A = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=G1[:,:3], hspace=0.2, wspace=0.4)
B = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=G1[:,-1])
C = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G2[0,1:], wspace=0.5)
D = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G2[1,1:], wspace=0.5)

ax_legend = fig.add_subplot(G2[0,0])
ax_legend.axis('off')

ax_A11 = fig.add_subplot(A[0,0])
ax_A11.text(-0.5, 1.25, 'A', transform=ax_A11.transAxes, fontsize=fs+1)
ax_A12 = fig.add_subplot(A[0,1], sharey=ax_A11)
plt.setp(ax_A12.get_yticklabels(), visible=False)
ax_A12.set_title('Neuromodulators acting on interneurons bias the weighting', fontsize=fs, pad=13)
ax_A13 = fig.add_subplot(A[0,2], sharey=ax_A11)
plt.setp(ax_A13.get_yticklabels(), visible=False)
ax_A21 = fig.add_subplot(A[1,0])
ax_A22 = fig.add_subplot(A[1,1], sharey=ax_A21)
plt.setp(ax_A22.get_yticklabels(), visible=False)
ax_A23 = fig.add_subplot(A[1,2], sharey=ax_A21)
plt.setp(ax_A23.get_yticklabels(), visible=False)

ax_B1 = fig.add_subplot(B[0,0], sharey=ax_A11)
ax_B1.text(-0.3, 1.25, 'B', transform=ax_B1.transAxes, fontsize=fs+1)
ax_B1.set_title('Effect can cancel out', fontsize=fs, pad=13)
plt.setp(ax_B1.get_yticklabels(), visible=False)
ax_B2 = fig.add_subplot(B[1,0], sharey=ax_A21)
plt.setp(ax_B2.get_yticklabels(), visible=False)

ax_C1 = fig.add_subplot(C[0,0])
ax_C1.text(-0.4, 1.25, 'C', transform=ax_C1.transAxes, fontsize=fs+1)
ax_C2 = fig.add_subplot(C[0,1])

ax_D1 = fig.add_subplot(D[0,0])
ax_D1.text(-0.4, 1.25, 'D', transform=ax_D1.transAxes, fontsize=fs+1)
ax_D2 = fig.add_subplot(D[0,1])

ax_CD = fig.add_subplot(G2[:,1:])
ax_CD.axis('off')
ax_CD.set_title(r'Sensory weight $\longleftarrow$ variance neuron $\longleftarrow$ PE neurons $\longleftarrow$ interneurons', fontsize=fs, pad=13)


# %% Neuromodulator acting on PV
   
xp, xs, xv = 1, 0, 0

file_for_data = '../results/data/neuromod/data_neuromod_' + str(xp) + '_' + str(xs) + '_' + str(xv) + '.pickle'
with open(file_for_data,'rb') as f:
    [xp, xs, xv, pert_strength, alpha] = pickle.load(f)

plot_neuromod_impact(pert_strength, alpha, xp, xs, xv, ax1=ax_A11, ax2=ax_A21, show_xlabel=False)


# %% Neuromodulator acting on SOM
  
xp, xs, xv = 0, 1, 0

file_for_data = '../results/data/neuromod/data_neuromod_' + str(xp) + '_' + str(xs) + '_' + str(xv) + '.pickle'
with open(file_for_data,'rb') as f:
    [xp, xs, xv, pert_strength, alpha] = pickle.load(f)

plot_neuromod_impact(pert_strength, alpha, xp, xs, xv, ax1=ax_A12, ax2=ax_A22, show_ylabel=False)


# %% Neuromodulator acting on VIP
  
xp, xs, xv = 0, 0, 1

file_for_data = '../results/data/neuromod/data_neuromod_' + str(xp) + '_' + str(xs) + '_' + str(xv) + '.pickle'
with open(file_for_data,'rb') as f:
    [xp, xs, xv, pert_strength, alpha] = pickle.load(f)

plot_neuromod_impact(pert_strength, alpha, xp, xs, xv, ax1=ax_A13, ax2=ax_A23, show_ylabel=False, show_xlabel=False)


# %% Neuromodulator acting on SOM & VIP
  
xp, xs, xv = 0, 0.5, 0.5

file_for_data = '../results/data/neuromod/data_neuromod_' + str(xp) + '_' + str(xs) + '_' + str(xv) + '.pickle'
with open(file_for_data,'rb') as f:
    [xp, xs, xv, pert_strength, alpha] = pickle.load(f)

plot_neuromod_impact(pert_strength, alpha, xp, xs, xv, ax1=ax_B1, ax2=ax_B2, show_ylabel=False)


# %%  plot legend

markers = ['o', 's', 'd']
label_text = ['SOM & PV', 'VIP & PV', 'SOM, VIP\nand PV']

for i in range(3):
    ax_legend.plot(np.nan,np.nan, marker=markers[i], color='k', label=label_text[i], ms=2.5, ls='None')
    
ax_legend.legend(loc=2, fontsize=fs, frameon=False, bbox_to_anchor=(-0.6, 0.), 
                    title='Sens. input to', title_fontsize=fs)

plot_standalone_colorbar(ax_legend)

# %% How are the vv neurons influenced by changes in nPE and pPE neurons in the lower and higher PE circuit?

plot_changes_upon_input2PE_neurons_new(ax1 = ax_C1, ax2 = ax_C2)


# %% How do the interneurons influence the PE neuron properties?

plot_influence_interneurons_baseline_or_gain(ax=ax_D1, plot_annotation=False)
plot_influence_interneurons_baseline_or_gain(plot_baseline=False, plot_annotation=False, ax=ax_D2)

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
