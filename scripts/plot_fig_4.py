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
import seaborn as sns

from src.plot_data import plot_neuromod_impact, plot_standalone_colorbar, plot_illustration_neuromod_results
from src.plot_data import plot_changes_upon_input2PE_neurons_new, plot_influence_interneurons_baseline_or_gain
from src.functions_simulate import stimuli_moments_from_uniform

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

G = gridspec.GridSpec(2, 1, figure=fig, hspace=0.4)

P1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=G[0,0], width_ratios=[1,4], wspace=0.3, hspace=0.3)
P2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=G[1,0], width_ratios=[1.5,1], wspace=0.7)

A = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=P1[0,1])
B = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=P1[1,1])
C = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=P2[0,0])
D = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=P2[0,1], hspace=0.6)

ax_A1 = fig.add_subplot(P1[0,0])
ax_A1.text(-0.3, 1.05, 'A', transform=ax_A1.transAxes, fontsize=fs+1)
ax_A2 = fig.add_subplot(A[0,0])
ax_A3 = fig.add_subplot(A[0,1], sharey=ax_A2)
ax_A4 = fig.add_subplot(A[0,2], sharey=ax_A2)
#ax_A1.axis('off')

ax_B1 = fig.add_subplot(P1[1,0])
ax_B1.text(-0.3, 1.05, 'B', transform=ax_B1.transAxes, fontsize=fs+1)
ax_B2 = fig.add_subplot(B[0,0])
ax_B3 = fig.add_subplot(B[0,1], sharey=ax_B2)
ax_B4 = fig.add_subplot(B[0,2], sharey=ax_B3)
#ax_B1.axis('off')

plt.setp(ax_A3.get_yticklabels(), visible=False)
plt.setp(ax_A4.get_yticklabels(), visible=False)
plt.setp(ax_B3.get_yticklabels(), visible=False)
plt.setp(ax_B4.get_yticklabels(), visible=False)
plt.setp(ax_A2.get_xticklabels(), visible=False)
plt.setp(ax_A3.get_xticklabels(), visible=False)
plt.setp(ax_A4.get_xticklabels(), visible=False)

ax_C = fig.add_subplot(C[0,0])
ax_C.axis('off')
ax_C.text(-0.15, 1.05, 'C', transform=ax_C.transAxes, fontsize=fs+1)

ax_D1 = fig.add_subplot(D[0,0])
ax_D2 = fig.add_subplot(D[1,0])
ax_D1.text(-0.6, 1.05, 'D', transform=ax_D1.transAxes, fontsize=fs+1)

ax_A = np.array([ax_A2, ax_A3, ax_A4])
ax_B = np.array([ax_B2, ax_B3, ax_B4])

# ax_CD.set_title(r'Sensory weight $\longleftarrow$ variance neuron $\longleftarrow$ PE neurons $\longleftarrow$ interneurons', fontsize=fs, pad=13)


# %% Plot stimuli

stimuli = stimuli_moments_from_uniform(11, 10, 5 - np.sqrt(3), 5 + np.sqrt(3), 0, 0)
stimuli = np.repeat(stimuli, 500)

ax_A1.plot(stimuli, color='#FEDFC2', lw=1, marker='|', ls="None")
for i in range(6):
    ax_A1.axvspan(2*i*5000, (2*i+1)*5000, color='#F5F4F5', zorder=0)

ax_A1.set_xticks([])
ax_A1.set_yticks([])
sns.despine(ax=ax_A1, top=True, left=True, right=True, bottom=True)   

ax_A1.set_ylabel('Sensory diven', fontsize=fs)

stimuli = stimuli_moments_from_uniform(11, 10, 5, 5, 0, 1)
stimuli = np.repeat(stimuli, 500)

ax_B1.plot(stimuli, color='#FEDFC2', lw=1, marker='|', ls="None")
for i in range(6):
    ax_B1.axvspan(2*i*5000, (2*i+1)*5000, color='#F5F4F5', zorder=0)

ax_B1.set_xticks([])
ax_B1.set_yticks([])
sns.despine(ax=ax_B1, top=True, left=True, right=True, bottom=True)    

ax_B1.set_ylabel('Prediction diven', fontsize=fs)
ax_B1.set_xlabel('Time (# Trials)', fontsize=fs)

# %% Neuromodulator acting on PV
   

xp, xs, xv = 0, 0.5, 0.5
file_for_data = '../results/data/neuromod/data_neuromod_' + str(xp) + '_' + str(xs) + '_' + str(xv) + '.pickle'
with open(file_for_data,'rb') as f:
    [xp, xs, xv, pert_strength, alpha] = pickle.load(f)

plot_neuromod_impact(pert_strength, alpha, xp, xs, xv, ax1=ax_A, ax2=ax_B, show_xlabel=True, show_ylabel=True)


xp, xs, xv = 0, 0, 1
file_for_data = '../results/data/neuromod/data_neuromod_' + str(xp) + '_' + str(xs) + '_' + str(xv) + '.pickle'
with open(file_for_data,'rb') as f:
    [xp, xs, xv, pert_strength, alpha] = pickle.load(f)

plot_neuromod_impact(pert_strength, alpha, xp, xs, xv, ax1=ax_A, ax2=ax_B)


xp, xs, xv = 0, 1, 0
file_for_data = '../results/data/neuromod/data_neuromod_' + str(xp) + '_' + str(xs) + '_' + str(xv) + '.pickle'
with open(file_for_data,'rb') as f:
    [xp, xs, xv, pert_strength, alpha] = pickle.load(f)

plot_neuromod_impact(pert_strength, alpha, xp, xs, xv, ax1=ax_A, ax2=ax_B)


xp, xs, xv = 1, 0, 0
file_for_data = '../results/data/neuromod/data_neuromod_' + str(xp) + '_' + str(xs) + '_' + str(xv) + '.pickle'
with open(file_for_data,'rb') as f:
    [xp, xs, xv, pert_strength, alpha] = pickle.load(f)

plot_neuromod_impact(pert_strength, alpha, xp, xs, xv, ax1=ax_A, ax2=ax_B)

ax_B2.legend(loc=2, fontsize=fs, frameon=False, handlelength=1.5, ncol=1, borderaxespad=0)


# %% Illustrate main results

plot_illustration_neuromod_results(ax=ax_C)

# %% How do the interneurons influence the PE neuron properties?

plot_influence_interneurons_baseline_or_gain(ax=ax_D1, plot_annotation=False)
plot_influence_interneurons_baseline_or_gain(plot_baseline=False, plot_annotation=False, ax=ax_D2)

# %% save figure

plt.savefig(figPath + figure_name, bbox_inches='tight', transparent=True, dpi=600)
