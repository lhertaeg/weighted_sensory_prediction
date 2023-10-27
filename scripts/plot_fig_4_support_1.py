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

from src.plot_data import plot_neuromod_impact_inter

# %% Universal parameters

fs = 6
inch = 2.54


# %% Define files and paths

figure_name = 'Fig_4_S1.png'
figPath = '../results/figures/final/'

if not os.path.exists(figPath):
    os.mkdir(figPath)


# %% Define figure structure

figsize=(18/inch,14/inch)
fig = plt.figure(figsize=figsize)

G = gridspec.GridSpec(2, 1, figure=fig, hspace=0.6)

G1 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=G[0,0], hspace=0.3)
G2 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=G[1,0], hspace=0.3)

ax_A = fig.add_subplot(G1[:,:])
ax_A.axis('off')
ax_A.set_title('Neuromodulators acting on interneurons in lower PE circuit', fontsize=fs, pad=10)
ax_A.text(-0.1, 1.1, 'A', transform=ax_A.transAxes, fontsize=fs+1)

ax_A12 = fig.add_subplot(G1[0,0])
plt.setp(ax_A12.get_xticklabels(), visible=False)
ax_A13 = fig.add_subplot(G1[0,1])
plt.setp(ax_A13.get_yticklabels(), visible=False)
plt.setp(ax_A13.get_xticklabels(), visible=False)
ax_A14 = fig.add_subplot(G1[0,2])
plt.setp(ax_A14.get_yticklabels(), visible=False)
plt.setp(ax_A14.get_xticklabels(), visible=False)

ax_A22 = fig.add_subplot(G1[1,0])
ax_A23 = fig.add_subplot(G1[1,1])
plt.setp(ax_A23.get_yticklabels(), visible=False)
ax_A24 = fig.add_subplot(G1[1,2])
plt.setp(ax_A24.get_yticklabels(), visible=False)

ax_B = fig.add_subplot(G2[:,:])
ax_B.axis('off')
ax_B.set_title('Neuromodulators acting on interneurons in higher PE circuit', fontsize=fs, pad=10)
ax_B.text(-0.1, 1.1, 'B', transform=ax_B.transAxes, fontsize=fs+1)

ax_B12 = fig.add_subplot(G2[0,0])
plt.setp(ax_B12.get_xticklabels(), visible=False)
ax_B13 = fig.add_subplot(G2[0,1])
plt.setp(ax_B13.get_yticklabels(), visible=False)
plt.setp(ax_B13.get_xticklabels(), visible=False)
ax_B14 = fig.add_subplot(G2[0,2])
plt.setp(ax_B14.get_yticklabels(), visible=False)
plt.setp(ax_B14.get_xticklabels(), visible=False)

ax_B22 = fig.add_subplot(G2[1,0])
ax_B23 = fig.add_subplot(G2[1,1])
plt.setp(ax_B23.get_yticklabels(), visible=False)
ax_B24 = fig.add_subplot(G2[1,2])
plt.setp(ax_B24.get_yticklabels(), visible=False)

# %% Neuromodulators acting on interneurons in lower PE circuit for 2 limit cases
    
plot_neuromod_impact_inter(1, 1, 0, s=7, ax1=ax_A12, ax2=ax_A13, ax3=ax_A14, flg_plot_xlabel=False, 
                           flg_plot_bars=False, highlight=False)
plot_neuromod_impact_inter(1, 0, 1, s=7, ax1=ax_A22, ax2=ax_A23, ax3=ax_A24, flg_plot_xlabel=False, 
                           highlight=False)


# %% Neuromodulators acting on interneurons in higher PE circuit for 2 limit cases
    
plot_neuromod_impact_inter(2, 1, 0, s=7, ax1=ax_B12, ax2=ax_B13, ax3=ax_B14, flg_plot_xlabel=False, 
                           flg_plot_bars=False, highlight=False)
plot_neuromod_impact_inter(2, 0, 1, s=7, ax1=ax_B22, ax2=ax_B23, ax3=ax_B24, flg_plot_xlabel=False, 
                           highlight=False)

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)