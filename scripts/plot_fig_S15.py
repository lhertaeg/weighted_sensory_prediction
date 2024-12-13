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
import numpy as np

from src.plot_data import plot_neuromod_impact_inter, plot_illustration_neuromod_results


# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_S15.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(12/inch,10/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(1, 3, figure=fig, wspace=0.1)
A = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=G[0,0])
B = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=G[0,1])
C = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=G[0,2])

ax_A1 = fig.add_subplot(A[0,0])
ax_A2 = fig.add_subplot(A[1,0])
ax_A3 = fig.add_subplot(A[2,0])

ax_B1 = fig.add_subplot(B[0,0], sharey=ax_A1)
ax_B2 = fig.add_subplot(B[1,0], sharey=ax_A2)
ax_B3 = fig.add_subplot(B[2,0], sharey=ax_A3)

ax_C1 = fig.add_subplot(C[0,0], sharey=ax_A1)
ax_C2 = fig.add_subplot(C[1,0], sharey=ax_A2)
ax_C3 = fig.add_subplot(C[2,0], sharey=ax_A3)

plt.setp(ax_B1.get_yticklabels(), visible=False)
plt.setp(ax_B2.get_yticklabels(), visible=False)
plt.setp(ax_B3.get_yticklabels(), visible=False)
plt.setp(ax_C1.get_yticklabels(), visible=False)
plt.setp(ax_C2.get_yticklabels(), visible=False)
plt.setp(ax_C3.get_yticklabels(), visible=False)

axs = np.array([[ax_A1, ax_B1, ax_C1], [ax_A2, ax_B2, ax_C2], [ax_A3, ax_B3, ax_C3]])

# %% Neuromodulators acting on interneurons for 2 limit cases
    
plot_neuromod_impact_inter(1, 1, 0, axs=axs)#, flg_plot_xlabel=False, flg_plot_bars=False)
plot_neuromod_impact_inter(2, 1, 0, axs=axs)
plot_neuromod_impact_inter(0, 1, 0, axs=axs)

axs[0,0].legend(loc=0, frameon=False, handlelength=1, fontsize=fs)

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
