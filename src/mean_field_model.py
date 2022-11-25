#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 08:05:23 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import pickle

dtype = np.float32

# %% Notes/Todo's:
    
    # For the sake of simplicity, I have squared the activity of the PE neurons only for the computation of the variance,
    # in later implementations, that needs to be revised
    
    
# s = 0
# N = 100000
# tau = 100

# for i in range(N):
#     s += np.random.uniform(1,5) * np.exp((i-(N-1))/tau) / tau

# print(s)

# %% functions

def default_para(filename, baseline_activity = dtype([0, 0, 0, 0, 4, 4, 4, 4]), VS=1, VV=0):
    
    ### time constants
    tc_var_per_stim = dtype(20000) #dtype(100)
    tc_var_pred = dtype(200) #dtype(100)
    tau_pe = [dtype(60), dtype(2)]
    
    ### weights
    
    ## load data   
    with open(filename,'rb') as f:
        xopt, W, optimize_flag, _, _ = pickle.load(f)
   
    ## connectivity within PE circuits
    w_PE_to_PE = np.copy(W)
    v_PE_to_PE = np.copy(W)
    
    w_PE_to_PE[optimize_flag!=0] = xopt
    v_PE_to_PE[optimize_flag!=0] = xopt
    
    ## external background input (here I assume it is the same for both parts)
    r_target = baseline_activity
    fixed_input = (np.eye(8, dtype=dtype) - w_PE_to_PE) @ r_target
    
    ## Define connectivity between PE circuit and P
    w_PE_to_P = np.zeros((1,8), dtype=dtype)     
    w_PE_to_P[0,0] = -0.003          # nPE onto P
    w_PE_to_P[0,1] =  0.003          # pPE onto P
    
    w_P_to_PE = np.zeros((8,1), dtype=dtype)     
    w_P_to_PE[2:4,0] = dtype(1)     # onto dendrites
    w_P_to_PE[5,0] = dtype(1)       # onto PV neuron receiving prediction
    w_P_to_PE[6,0] = dtype(1-VS)    # onto SOM neuron
    w_P_to_PE[7,0] = dtype(1-VV)    # onto VIP neuron
    
    v_PE_to_P = np.zeros((1,8), dtype=dtype)     
    v_PE_to_P[0,0] = -1e-3          # nPE onto P
    v_PE_to_P[0,1] =  1e-3          # pPE onto P
    
    v_P_to_PE = np.zeros((8,1), dtype=dtype)     
    v_P_to_PE[2:4,0] = dtype(1)     # onto dendrites
    v_P_to_PE[5,0] = dtype(1)       # onto PV neuron receiving prediction
    v_P_to_PE[6,0] = dtype(1-VS)    # onto SOM neuron
    v_P_to_PE[7,0] = dtype(1-VV)    # onto VIP neuron
    
        
    return w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, tc_var_per_stim, tc_var_pred, tau_pe, fixed_input
        

def stimuli_moments_from_uniform(n_stimuli, stimulus_duration, min_mean, max_mean, min_std, max_std, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    mean_stimuli = np.random.uniform(min_mean, max_mean, size=n_stimuli)
    sd_stimuli = np.random.uniform(min_std, max_std, size=n_stimuli)
    stimuli = np.array([])
    
    for id_stim in range(n_stimuli):
        
        inputs_per_stimulus = np.random.normal(mean_stimuli[id_stim], sd_stimuli[id_stim], size=stimulus_duration)
        stimuli = np.concatenate((stimuli, inputs_per_stimulus))
        
    return stimuli



def pe_circuit_dynamics(tau_E, tau_I, W, rates_pe_circuit, feedback_input, feedforward_input, dt):
    
    # Initialise
    rates_pe_circuit_initial = rates_pe_circuit.copy() 
    dr_pe_1 = np.zeros(len(rates_pe_circuit), dtype=dtype)
    dr_pe_2 = np.zeros(len(rates_pe_circuit), dtype=dtype)
    
    # RK 2nd order
    dr_pe_1 = -rates_pe_circuit_initial + W @ rates_pe_circuit_initial + feedback_input + feedforward_input
    dr_pe_1[:4] /= tau_E 
    dr_pe_1[4:] /= tau_I 

    rates_pe_circuit_initial[:] += dt * dr_pe_1
    
    dr_pe_2 = -rates_pe_circuit_initial + W @ rates_pe_circuit_initial + feedback_input + feedforward_input
    dr_pe_2[:4] /= tau_E 
    dr_pe_2[4:] /= tau_I
    
    rates_pe_circuit += dt/2 * (dr_pe_1 + dr_pe_2)
    
    # Rectify
    rates_pe_circuit[rates_pe_circuit<0] = 0

    return rates_pe_circuit



def run_pe_circuit_mfn(w_PE_to_PE, tau_pe, fixed_input, stimuli, prediction, VS = 1, VV = 0, dt = dtype(1)):
    
    ### neuron and network parameters
    tau_E, tau_I  = tau_pe
    neurons_feedforward = np.array([1, 1, 0, 0, 1, 0, VS, VV], dtype=dtype)
    neurons_feedback = 1 - neurons_feedforward
    
    if fixed_input.ndim==1:
        n_stimuli = len(stimuli)
        fixed_input = np.tile(fixed_input,(n_stimuli,1))
    
    ### initialise
    rates_pe_circuit = np.zeros((len(stimuli), 8), dtype=dtype)
    
    ### run network
    for id_stim, stim in enumerate(stimuli):
              
        ## mean-field network, PE circuit (feedforward = sensory, fedback = prediction)
        feedforward_input = fixed_input[id_stim,:] + stim * neurons_feedforward
        feedback_input = prediction[id_stim] * neurons_feedback
        
        rates_pe_circuit[id_stim,:] = pe_circuit_dynamics(tau_E, tau_I, w_PE_to_PE, rates_pe_circuit[id_stim-1,:], 
                                                          feedback_input, feedforward_input, dt)
    return rates_pe_circuit



def rate_dynamics_mfn(tau_E, tau_I, U, V, W, rates_pe_circuit, rate_memory_neuron, feedforward_input, dt):
    
    # Initialise
    rates_pe_circuit_initial = rates_pe_circuit.copy() 
    rate_memory_neuron_initial = rate_memory_neuron.copy()
    dr_pe_1 = np.zeros(len(rates_pe_circuit), dtype=dtype)
    dr_pe_2 = np.zeros(len(rates_pe_circuit), dtype=dtype)
    dr_mem_1 = np.zeros((1,1), dtype=dtype)
    dr_mem_2 = np.zeros((1,1), dtype=dtype)
    
    # RK 2nd order
    # rates_pe_circuit_initial [:2] **= 2 (use later to make sure that prediction and var are computed though PE**2 ... not used so far because I use MF parameters that I have extracted from linear IO)
    dr_mem_1 =  (U @ rates_pe_circuit_initial) / tau_E
    dr_pe_1 = -rates_pe_circuit_initial + W @ rates_pe_circuit_initial + V @ np.array([rate_memory_neuron_initial]) + feedforward_input
    dr_pe_1[:4] /= tau_E 
    dr_pe_1[4:] /= tau_I 

    rates_pe_circuit_initial[:] += dt * dr_pe_1
    rate_memory_neuron_initial += dt * dr_mem_1
    
    dr_mem_2 = (U @ rates_pe_circuit_initial) / tau_E
    dr_pe_2 = -rates_pe_circuit_initial + W @ rates_pe_circuit_initial + V @ rate_memory_neuron_initial + feedforward_input
    dr_pe_2[:4] /= tau_E 
    dr_pe_2[4:] /= tau_I
    
    rates_pe_circuit += dt/2 * (dr_pe_1 + dr_pe_2)
    rate_memory_neuron += dt/2 * (dr_mem_1 + dr_mem_2)
    
    # Rectify
    rates_pe_circuit[rates_pe_circuit<0] = 0

    return rates_pe_circuit, rate_memory_neuron


def run_mean_field_model_one_column(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, fixed_input, 
                                    stimuli, VS = 1, VV = 0, dt = dtype(1), set_initial_prediction_to_mean = False, 
                                    set_initial_prediction_to_value = np.nan, w_PE_to_V = dtype([1,1])):
    
    ### neuron and network parameters
    tau_E, tau_I  = tau_pe
    neurons_feedforward = np.array([1, 1, 0, 0, 1, 0, VS, VV], dtype=dtype)
    
    ### initialise
    prediction = np.zeros_like(stimuli, dtype=dtype)
    variance_per_stimulus = np.zeros_like(stimuli, dtype=dtype)
    
    if set_initial_prediction_to_mean:
        prediction[-1] = np.mean(stimuli)
        
    if set_initial_prediction_to_value is not np.nan:
        prediction[-1] = set_initial_prediction_to_value
        
    rates_pe_circuit_sens = np.zeros((len(stimuli), 8), dtype=dtype)
    
    if fixed_input.ndim==1:
        n_stimuli = len(stimuli)
        fixed_input = np.tile(fixed_input,(n_stimuli,1))
    
    ### compute prediction-errors, prediction, mean of prediction, variance of sensory input, variance of prediction
    for id_stim, stim in enumerate(stimuli):
              
        ## mean-field network, PE circuit (feedforward = sensory, fedback = prediction)
        feedforward_input = fixed_input[id_stim,:] + stim * neurons_feedforward
        rates_pe_circuit_sens[id_stim,:], prediction[id_stim] = rate_dynamics_mfn(tau_E, tau_I, w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                                  rates_pe_circuit_sens[id_stim-1,:], prediction[id_stim-1], 
                                                                                  feedforward_input, dt)


        ## compute variance of sensory input and prediction
        nPE_sensory, pPE_sensory = (w_PE_to_V * rates_pe_circuit_sens[id_stim, :2]) 
        variance_per_stimulus[id_stim] = (1-1/tc_var_per_stim) * variance_per_stimulus[id_stim-1] + (nPE_sensory + pPE_sensory)**2/tc_var_per_stim
        # Note: 
            # as I assume that nPE and pPE neurons have zero BL activity, I can put the x**2 into the V neuron
            # please note that this might have to change for a large-scale network
            # as I need to make sure that nPE and pPE neurons have the same effect and represent S-P and P-S, respectively, I might need to add some scaling
    
    return prediction, variance_per_stimulus, rates_pe_circuit_sens[:,:2]


def run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                         tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli, 
                         VS = 1, VV = 0, dt = dtype(1), set_initial_prediction_to_mean = False,
                         w_PE_to_V = dtype([1,1]), v_PE_to_V = dtype([1,1]), 
                         fixed_input_1 = None, fixed_input_2 = None, PE=False):
    
    ### neuron and network parameters
    tau_E, tau_I  = tau_pe
    neurons_feedforward = np.array([1, 1, 0, 0, 1, 0, VS, VV], dtype=dtype)
    
    ### initialise
    nPE = np.zeros_like(stimuli, dtype=dtype)
    pPE = np.zeros_like(stimuli, dtype=dtype)
    prediction = np.zeros_like(stimuli, dtype=dtype)
    mean_pred = np.zeros_like(stimuli, dtype=dtype)   
    variance_per_stimulus = np.zeros_like(stimuli, dtype=dtype)
    variance_prediction = np.zeros_like(stimuli, dtype=dtype)
    
    if set_initial_prediction_to_mean:
        prediction[-1] = np.mean(stimuli)
        mean_pred[-1] = np.mean(stimuli)
    
    rates_pe_circuit_sens = np.zeros((len(stimuli), 8), dtype=dtype)
    rates_pe_circuit_pred = np.zeros((len(stimuli), 8), dtype=dtype)
    
    ### fixed input (external input and possible perturbation)
    if fixed_input is not None:
        
        if fixed_input.ndim==1:
            n_stimuli = len(stimuli)
            fixed_input = np.tile(fixed_input,(n_stimuli,1))
            
        fixed_input_1 = fixed_input
        fixed_input_2 = fixed_input
        
    else:
        
        if fixed_input_1.ndim==1:
            n_stimuli = len(stimuli)
            fixed_input_1 = np.tile(fixed_input_1,(n_stimuli,1))
            
        if fixed_input_2.ndim==1:
            n_stimuli = len(stimuli)
            fixed_input_2 = np.tile(fixed_input_2,(n_stimuli,1))
        
    ### compute prediction-errors, prediction, mean of prediction, variance of sensory onput, variance of prediction
    for id_stim, stim in enumerate(stimuli):
              
        ## mean-field network, PE circuit (feedforward = sensory, fedback = prediction)
        feedforward_input = fixed_input_1[id_stim,:] + stim * neurons_feedforward
        rates_pe_circuit_sens[id_stim,:], prediction[id_stim] = rate_dynamics_mfn(tau_E, tau_I, w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                                  rates_pe_circuit_sens[id_stim-1,:], prediction[id_stim-1], 
                                                                                  feedforward_input, dt)
        
        ## mean-field network, PE circuit (feedforward = prediction, fedback = prediction of prediction)
        feedforward_input = fixed_input_2[id_stim,:] + prediction[id_stim-1] * neurons_feedforward
        rates_pe_circuit_pred[id_stim,:], mean_pred[id_stim] = rate_dynamics_mfn(tau_E, tau_I, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                                 rates_pe_circuit_pred[id_stim-1,:], mean_pred[id_stim-1], 
                                                                                 feedforward_input, dt)

        ## compute variance of sensory input and prediction
        # please note that dt = 1 here!
        nPE_sensory, pPE_sensory = (w_PE_to_V * rates_pe_circuit_sens[id_stim, :2]) 
        variance_per_stimulus[id_stim] = (1-1/tc_var_per_stim) * variance_per_stimulus[id_stim-1] + (nPE_sensory + pPE_sensory)**2/tc_var_per_stim

        nPE_prediction, pPE_prediction = (v_PE_to_V * rates_pe_circuit_pred[id_stim, :2]) 
        variance_prediction[id_stim] = (1-1/tc_var_pred) * variance_prediction[id_stim-1] + (nPE_prediction + pPE_prediction)**2/tc_var_pred
        
        nPE[id_stim] = nPE_sensory
        pPE[id_stim] = pPE_sensory

    ### compute weighted output
    variance_per_stimulus[np.isinf(1/variance_per_stimulus)] = 1e-30
    variance_prediction[np.isinf(1/variance_prediction)] = 1e-30
    
    alpha = (1/variance_per_stimulus) / ((1/variance_per_stimulus) + (1/variance_prediction))
    beta = (1/variance_prediction) / ((1/variance_per_stimulus) + (1/variance_prediction))
   
    weighted_output = alpha * stimuli + beta * prediction
    
    ret = (prediction, variance_per_stimulus, mean_pred, variance_prediction, alpha, beta, weighted_output,)
    if PE:
        ret += (nPE, pPE, )
    
    return ret #prediction, variance_per_stimulus, mean_pred, variance_prediction, alpha, beta, weighted_output


def run_mean_field_model_pred(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                              tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli, 
                              VS = 1, VV = 0, dt = dtype(1), set_initial_prediction_to_mean = False,
                              w_PE_to_V = dtype([1,1]), v_PE_to_V = dtype([1,1])):
    
    ### neuron and network parameters
    tau_E, tau_I  = tau_pe
    neurons_feedforward = np.array([1, 1, 0, 0, 1, 0, VS, VV], dtype=dtype)
    
    ### initialise
    prediction = np.zeros_like(stimuli, dtype=dtype)
    mean_pred = np.zeros_like(stimuli, dtype=dtype)   
    variance_per_stimulus = np.zeros_like(stimuli, dtype=dtype)
    variance_prediction = np.zeros_like(stimuli, dtype=dtype)
    
    if set_initial_prediction_to_mean:
        prediction[-1] = np.mean(stimuli)
        mean_pred[-1] = np.mean(stimuli)
    
    rates_pe_circuit_sens = np.zeros((len(stimuli), 8), dtype=dtype)
    rates_pe_circuit_pred = np.zeros((len(stimuli), 8), dtype=dtype)
    
    ### compute prediction-errors, prediction, mean of prediction, variance of sensory onput, variance of prediction
    for id_stim, stim in enumerate(stimuli):
              
        ## mean-field network, PE circuit (feedforward = sensory, fedback = prediction)
        feedforward_input = fixed_input + stim * neurons_feedforward
        rates_pe_circuit_sens[id_stim,:], prediction[id_stim] = rate_dynamics_mfn(tau_E, tau_I, w_PE_to_P, w_P_to_PE, w_PE_to_PE, 
                                                                                  rates_pe_circuit_sens[id_stim-1,:], prediction[id_stim-1], 
                                                                                  feedforward_input, dt)
        
        ## mean-field network, PE circuit (feedforward = prediction, fedback = prediction of prediction)
        feedforward_input = fixed_input + prediction[id_stim-1] * neurons_feedforward
        rates_pe_circuit_pred[id_stim,:], mean_pred[id_stim] = rate_dynamics_mfn(tau_E, tau_I, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                                                 rates_pe_circuit_pred[id_stim-1,:], mean_pred[id_stim-1], 
                                                                                 feedforward_input, dt)

        ## compute variance of sensory input and prediction
        nPE_sensory, pPE_sensory = (w_PE_to_V * rates_pe_circuit_sens[id_stim, :2]) 
        variance_per_stimulus[id_stim] = (1-1/tc_var_per_stim) * variance_per_stimulus[id_stim-1] + (nPE_sensory + pPE_sensory)**2/tc_var_per_stim

        nPE_prediction, pPE_prediction = (v_PE_to_V * rates_pe_circuit_pred[id_stim, :2]) 
        variance_prediction[id_stim] = (1-1/tc_var_pred) * variance_prediction[id_stim-1] + (nPE_prediction + pPE_prediction)**2/tc_var_pred

    ### compute weighted output
    variance_per_stimulus[np.isinf(1/variance_per_stimulus)] = 1e-30
    variance_prediction[np.isinf(1/variance_prediction)] = 1e-30
    
    alpha = (1/variance_per_stimulus) / ((1/variance_per_stimulus) + (1/variance_prediction))
    beta = (1/variance_prediction) / ((1/variance_per_stimulus) + (1/variance_prediction))
    
    weighted_prediction = alpha * prediction + beta * mean_pred
    
    return prediction, variance_per_stimulus, mean_pred, variance_prediction, alpha, beta, weighted_prediction



def alpha_parameter_exploration(para_tested_first, para_tested_second, w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                tc_var_pred, tau_pe, fixed_input, stimuli, stimulus_duration, n_stimuli, last_n, para_exploration_name):
    
    ### number of parameters and stimuli tested
    num_para_first = len(para_tested_first)
    num_para_second = len(para_tested_second)
    
    ### initialise
    fraction_sensory_mean = np.zeros((num_para_first, num_para_second), dtype=dtype)
    fraction_sensory_median = np.zeros((num_para_first, num_para_second), dtype=dtype)
    fraction_sensory_std = np.zeros((num_para_first, num_para_second), dtype=dtype)
    
    ### run networks over all parameter to be tested
    if (para_exploration_name=='tc'):
        
        for row, para_first in enumerate(para_tested_first):
            for col, para_second in enumerate(para_tested_second):
                
                _, _, _, _, alpha, _, _ = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                               para_first, para_second, tau_pe, fixed_input, stimuli, set_initial_prediction_to_mean=True)
                
                fraction_sensory_median[row, col] = np.median(alpha[(n_stimuli - last_n) * stimulus_duration:])
                fraction_sensory_mean[row, col] = np.mean(alpha[(n_stimuli - last_n) * stimulus_duration:])
                fraction_sensory_std[row, col] = np.std(alpha[(n_stimuli - last_n) * stimulus_duration:])
    
    elif para_exploration_name=='w':
        
        for row, para_first in enumerate(para_tested_first):
            for col, para_second in enumerate(para_tested_second):
                
                # para first
                w_PE_to_P[0,0] = -para_first         
                w_PE_to_P[0,1] =  para_first 
                
                # para second
                v_PE_to_P[0,0] = -para_second
                v_PE_to_P[0,1] =  para_second
                
                _, _, _, _, alpha, _, _ = run_mean_field_model(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, v_P_to_PE, v_PE_to_PE, 
                                                               tc_var_per_stim, tc_var_pred, tau_pe, fixed_input, stimuli, set_initial_prediction_to_mean=True)
                
                fraction_sensory_median[row, col] = np.median(alpha[(n_stimuli - last_n) * stimulus_duration:])
                fraction_sensory_mean[row, col] = np.mean(alpha[(n_stimuli - last_n) * stimulus_duration:])
                fraction_sensory_std[row, col] = np.std(alpha[(n_stimuli - last_n) * stimulus_duration:])
                
   
    return fraction_sensory_mean, fraction_sensory_median, fraction_sensory_std
