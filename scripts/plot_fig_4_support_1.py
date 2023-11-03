#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:04:01 2023

@author: loreen.hertaeg
"""

# %% import

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path

from src.plot_data import plot_neuromod_impact_inter, plot_illustration_neuromod_results


# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_4_S1.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(14/inch,14/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(2, 1, figure=fig, hspace=0.7)
A = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=G[0,0])
B = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=G[1,0])

ax_A = fig.add_subplot(A[:,:4])
ax_A.axis('off')
ax_A.set_title('Neuromodulators acting on interneurons can bias the weighting of sensory inputs and predictions', fontsize=fs, pad=18)

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

ax_B = fig.add_subplot(B[0,1:-1])


# %% Neuromodulators acting on interneurons for 2 limit cases
    
plot_neuromod_impact_inter(0, 1, 0, s=7, ax1=ax_A12, ax2=ax_A13, ax3=ax_A14, flg_plot_xlabel=False, flg_plot_bars=False)
plot_neuromod_impact_inter(0, 0, 1, s=7, ax1=ax_A22, ax2=ax_A23, ax3=ax_A24, flg_plot_xlabel=False)
 

# %% Illustrate main results

plot_illustration_neuromod_results(ax=ax_B)


# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
