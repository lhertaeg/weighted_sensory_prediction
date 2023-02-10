#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:31:58 2023

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

dtype = np.float32

# %% functions


def rate_dynamics_mfn(tau_E, tau_I, tc_var, w_var, U, V, W, rates, mean, 
                      var, feedforward_input, dt, n):
    
    # Initialise
    rates_new = rates.copy() 
    mean_new = mean.copy()
    dr_1 = np.zeros(len(rates_new), dtype=dtype)
    dr_2 = np.zeros(len(rates_new), dtype=dtype)
    dr_mem_1 = np.zeros((1,1), dtype=dtype)
    dr_mem_2 = np.zeros((1,1), dtype=dtype)
    
    var_new = var.copy() 
    dr_var_1 = np.zeros((1,1), dtype=dtype) 
    dr_var_2 = np.zeros((1,1), dtype=dtype) 
    
    # RK 2nd order
    dr_mem_1 = (U @ rates_new) / tau_E
    dr_var_1 = (-var_new + sum(w_var * rates_new[:2])**n) / tc_var 
    dr_1 = -rates_new + W @ rates_new + V @ np.array([mean_new]) + feedforward_input
    dr_1[:4] /= tau_E 
    dr_1[4:] /= tau_I 

    rates_new[:] += dt * dr_1
    mean_new += dt * dr_mem_1
    var_new += dt * dr_var_1
    
    dr_mem_2 = (U @ rates_new) / tau_E
    dr_var_2 = (-var_new + sum(w_var * rates_new[:2])**n) / tc_var
    dr_2 = -rates_new + W @ rates_new + V @ mean_new + feedforward_input
    dr_2[:4] /= tau_E 
    dr_2[4:] /= tau_I
    
    rates[:] += dt/2 * (dr_1 + dr_2)
    mean += dt/2 * (dr_mem_1 + dr_mem_2)
    var += dt/2 * (dr_var_1 + dr_var_2)
    
    # Rectify
    rates[rates<0] = 0

    return [rates, mean, var]


def run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, fixed_input, 
                        stimuli, VS = 1, VV = 0, dt = dtype(1), w_PE_to_V = dtype([1,1]), n=2):
    
    ### neuron and network parameters
    tau_E, tau_I  = tau_pe
    neurons_feedforward = np.array([1, 1, 0, 0, 1, 0, VS, VV], dtype=dtype)
    
    ### initialise
    num_points = len(stimuli)
    m_neuron = np.zeros_like(stimuli, dtype=dtype)
    v_neuron = np.zeros_like(stimuli, dtype=dtype)
    rates_lower = np.zeros((num_points, 8), dtype=dtype)
    
    if fixed_input.ndim==1:
        fixed_input = np.tile(fixed_input, (num_points,1))
    
    ### run mean-field network
    for id_stim, stim in enumerate(stimuli):
        
        feedforward_input = fixed_input[id_stim,:] + stim * neurons_feedforward
        
        ## rates of PE circuit and M neuron
        [rates_lower[id_stim,:], 
         m_neuron[id_stim], v_neuron[id_stim]] = rate_dynamics_mfn(tau_E, tau_I, tc_var_per_stim, w_PE_to_V, 
                                                                   w_PE_to_P, w_P_to_PE, w_PE_to_PE,
                                                                   rates_lower[id_stim-1,:], m_neuron[id_stim-1], 
                                                                   v_neuron[id_stim-1], feedforward_input, dt, n)

    return m_neuron, v_neuron, rates_lower[:,:2]


def run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                            tc_var_pred, tau_pe, fixed_input, stimuli, VS = 1, VV = 0, dt = dtype(1), n = 2, 
                            w_PE_to_V = dtype([1,1]), v_PE_to_V  = dtype([1,1]), record_pe_activity = False,
                            fixed_input_lower = None, fixed_input_higher = None):
    
    ### neuron and network parameters
    tau_E, tau_I  = tau_pe
    neurons_feedforward = np.array([1, 1, 0, 0, 1, 0, VS, VV], dtype=dtype)
    
    ### initialise
    num_points = len(stimuli)
    m_neuron_lower = np.zeros_like(stimuli, dtype=dtype)
    v_neuron_lower = np.zeros_like(stimuli, dtype=dtype)
    m_neuron_higher = np.zeros_like(stimuli, dtype=dtype)
    v_neuron_higher = np.zeros_like(stimuli, dtype=dtype)
    rates_lower = np.zeros((num_points, 8), dtype=dtype)
    rates_higher = np.zeros((num_points, 8), dtype=dtype)
    
    if fixed_input is not None:
        if fixed_input.ndim==1:
            fixed_input_lower = np.tile(fixed_input, (num_points,1))
            fixed_input_higher = np.tile(fixed_input, (num_points,1))
        else:
            fixed_input_lower = fixed_input
            fixed_input_higher = fixed_input
    else:
        if fixed_input_lower.ndim==1:
            fixed_input_lower = np.tile(fixed_input_lower, (num_points,1))
        if fixed_input_higher.ndim==1:
            fixed_input_higher = np.tile(fixed_input_higher, (num_points,1))
            
    
    ### run mean-field network
    for id_stim, stim in enumerate(stimuli):
        
        ## run lower PE circuit
        feedforward_input_lower = fixed_input_lower[id_stim,:] + stim * neurons_feedforward
        
        [rates_lower[id_stim,:], m_neuron_lower[id_stim], 
         v_neuron_lower[id_stim]] = rate_dynamics_mfn(tau_E, tau_I, tc_var_per_stim, w_PE_to_V, w_PE_to_P,
                                                      w_P_to_PE, w_PE_to_PE, rates_lower[id_stim-1,:],
                                                      m_neuron_lower[id_stim-1], v_neuron_lower[id_stim-1], 
                                                     feedforward_input_lower, dt, n)
        
        
        ## run higher PE circuit
        feedforward_input_higher = fixed_input_higher[id_stim,:] + m_neuron_lower[id_stim-1] * neurons_feedforward
        
        [rates_higher[id_stim,:], m_neuron_higher[id_stim], 
         v_neuron_higher[id_stim]] = rate_dynamics_mfn(tau_E, tau_I, tc_var_pred, v_PE_to_V, v_PE_to_P,
                                                      v_P_to_PE, v_PE_to_PE, rates_higher[id_stim-1,:],
                                                      m_neuron_higher[id_stim-1], v_neuron_higher[id_stim-1], 
                                                      feedforward_input_higher, dt, n)
                                                       
    ### compute weighted output
    v_neuron_lower[np.isinf(1/v_neuron_lower)] = 1e-30
    v_neuron_higher[np.isinf(1/v_neuron_higher)] = 1e-30
    
    alpha = (1/v_neuron_lower) / ((1/v_neuron_lower) + (1/v_neuron_higher))
    beta = (1/v_neuron_higher) / ((1/v_neuron_lower) + (1/v_neuron_higher))
   
    weighted_output = alpha * stimuli + beta * m_neuron_lower
    
    ret = (m_neuron_lower, v_neuron_lower, m_neuron_higher, v_neuron_higher, alpha, beta, weighted_output,)
    
    if record_pe_activity:
        ret += (rates_lower[:,:2], rates_higher[:,:2], )
    
    return ret

    