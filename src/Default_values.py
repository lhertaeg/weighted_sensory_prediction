#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:11:50 2022

@author: loreen.hertaeg
"""

import numpy as np

dtype = np.float32

# %% Define default parameters

def Default_NeuronPara():
    
    NumCells = np.array([140,20,20,20], dtype=np.int32)
    tau_E_inv = dtype(1/60)
    tau_I_inv = dtype(1/2)
    
    return NumCells, tau_E_inv, tau_I_inv


def Default_Connectivity(wED: dtype = dtype(1), wEP: dtype = dtype(-2), 
                         wDE: dtype = dtype(0.5), wDS: dtype = dtype(-0.5), 
                         wPE: dtype = dtype(1.2), wPP: dtype = dtype(-1),  
                         wPS: dtype = dtype(-0.3), wPV: dtype = dtype(-0.3), 
                         wSV: dtype = dtype(-0.6), wVS: dtype = dtype(-0.7)): 
                         
                         
    weights_mean = np.array([[0,wED,wEP,0,0],[wDE,0,0,wDS,0],[wPE,0,wPP,wPS,wPV],
                             [1,0,0,0,wSV],[1,0,0,wVS,0]], dtype=dtype)                   
    
    conn_prob = np.array([[0,1,0.6,0,0],[0.1,0,0,0.55,0],[0.45,0,0.5,0.6,0.5],
                          [0.35,0,0,0,0.5],[0.1,0,0,0.45,0]], dtype=dtype)
    
    weight_name = np.array([['wEE','wED','wEP','wES','wEV'],['wDE','wDD','wDP','wDS','wDV'],['wPE','wPD','wPP','wPS','wPV'],
                            ['wSE','wSD','wSP','wSS','wSV'],['wVE','wVD','wVP','wVS','wVV']])

    temp_name = np.array([['TEE','TED','TEP','TES','TEV'],['TDE','TDD','TDP','TDS','TDV'],['TPE','TPD','TPP','TPS','TPV'],
                          ['TSE','TSD','TSP','TSS','TSV'],['TVE','TVD','TVP','TVS','TVV']])

    return weights_mean, conn_prob, weight_name, temp_name


def Default_LearningRates(eEP: dtype = dtype(1e-3), eDS: dtype = dtype(1e-4), 
                          ePS: dtype = dtype(1e-3), ePV: dtype = dtype(1e-3)): # ePS: dtype = dtype(1e-4), ePV: dtype = dtype(1e-4)
    
    eta_mean = np.array([[0,0,eEP,0,0],[0,0,0,eDS,0],[0,0,0,ePS,ePV],
                         [0,0,0,0,0],[0,0,0,0,0]], dtype=dtype)
    
    eta_name = np.array([['etaEE','etaED','etaEP','etaES','etaEV'],
                         ['etaDE','etaDD','etaDP','etaDS','etaDV'],
                         ['etaPE','etaPD','etaPP','etaPS','etaPV'],
                         ['etaSE','etaSD','etaSP','etaSS','etaSV'],
                         ['etaVE','etaVD','etaVP','etaVS','etaVV']])

    return eta_mean, eta_name


def Default_PredProcPara():
    
    m_out = dtype(0.5)
    mLE = dtype(1e-2)
    
    w_pred = dtype(0.5)
    w_classic = dtype(0.5)
    eta = dtype(5e-4)
    
    stimuli_weak = np.linspace(1,5,5)
    stimuli_strong = np.linspace(5,9,5)
    
    return m_out, mLE, w_pred, w_classic, eta, stimuli_weak, stimuli_strong


# def DefaultParameters_within():
    
#     num_cells_areas, 
#     conn_prob_within, 
#     indegree_within, 
#     digagonal_within, 
#     total_weight_within, 
#     mean_weight_within,
#     sd_weight_within, 
#     min_weight_within, 
#     max_weight_within, 
#     weight_dist_within, 
#     weight_name_within, 
#     temp_name_within


# def DefaultParameters_between():
    
#     num_cells_region_pre, 
#     num_cells_region_post, 
#     conn_prob_between, 
#     indegree_between, 
#     digagonal_between, 
#     total_weight_between, 
#     mean_weight_between, 
#     sd_weight_between, 
#     min_weight_between, 
#     max_weight_between, 
#     weight_dist_between, 
#     weight_name_between, 
#     temp_name_between