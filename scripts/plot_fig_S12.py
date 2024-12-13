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
import os.path

from src.plot_data import plot_illustration_changes_upon_baseline_PE, plot_illustration_changes_upon_gain_PE, plot_legend_illustrations

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_S12.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(10/inch,8/inch)
fig = plt.figure(figsize=figsize)


G = gridspec.GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.2)

ax_A = fig.add_subplot(G[:,:2])
ax_A.axis('off')
ax_A.set_title('Illustration of changes in mean and variance \nwith changes in baseline (left) and gain (right)', fontsize=fs, pad=10)

ax_A1 = fig.add_subplot(G[0,0])
ax_A2 = fig.add_subplot(G[1,0])
ax_A3 = fig.add_subplot(G[2,0])

ax_B1 = fig.add_subplot(G[0,1])
ax_B2 = fig.add_subplot(G[1,1])
ax_B3 = fig.add_subplot(G[2,1])

ax_legend_M = fig.add_subplot(G[0,-1])
ax_legend_M.axis('off')

ax_legend_V = fig.add_subplot(G[2,-1])
ax_legend_V.axis('off')

# %%  How is the variance influenced by BL of nPE and pPE in lower and higher PE circuit?    
 
plot_illustration_changes_upon_baseline_PE(BL = np.linspace(0,3,7), ax1 = ax_A1, ax2 = ax_A2, ax3 = ax_A3)

# %%  How is the variance influenced by gain of nPE and pPE in lower and higher PE circuit?    
 
plot_illustration_changes_upon_gain_PE(gains=np.linspace(0.5,1.5,7), ax1 = ax_B1, ax2 = ax_B2, ax3 = ax_B3)
plot_legend_illustrations(ax_legend_M, plot_M = True)
plot_legend_illustrations(ax_legend_V, plot_M = False)

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)