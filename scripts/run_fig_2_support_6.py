#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:50:33 2023

@author: loreen.hertaeg
"""

# %% Import

import pickle
import numpy as np

from src.functions_simulate import stimuli_moments_from_uniform, random_uniform_from_moments, simulate_example_pe_circuit, simulate_effect_baseline
from src.default_parameters import default_para_mfn
from src.functions_networks import run_mfn_circuit, run_mfn_circuit_coupled

dtype = np.float32


# %% Universal settings
    
### choose mean-field network to simulate
mfn_flag = '10' # valid options are '10', '01', '11

### load default parameters
VS, VV = int(mfn_flag[0]), int(mfn_flag[1])

### generate stimuli
np.random.seed(186)
trial_duration_1 = np.int32(150000)
num_values_per_trial_1 = np.int32(300)

repeats_per_value = trial_duration_1//num_values_per_trial_1

stimuli_1 = random_uniform_from_moments(5, 2, num_values_per_trial_1)
stimuli_1 = np.repeat(stimuli_1, repeats_per_value)


n_trials = np.int32(150)
trial_duration_2 = np.int32(5000)
num_values_per_trial_2 = np.int32(10)
np.random.seed(186)

n_repeats_per_stim = dtype(trial_duration_2/num_values_per_trial_2)

stimuli_2 = stimuli_moments_from_uniform(n_trials, num_values_per_trial_2, dtype(5 - np.sqrt(3)*5), 
                                       dtype(5 + np.sqrt(3)*5), dtype(0), dtype(5))
stimuli_2 = np.repeat(stimuli_2, n_repeats_per_stim)


# %% Robustness: tau_V

run_cell = False

if run_cell:
    
    ### filename for data
    file_for_data = '../results/data/moments/data_change_tauV.pickle'
    
    ## parameters tested
    paras = [1000, 9000]
    
    ### one column
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
    
    ## run network - parameter smaller 
    tc_var_per_stim = dtype(paras[0])
    m_neuron_s, v_neuron_s, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                    fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
    
    ## run network - parameter larger
    tc_var_per_stim = dtype(paras[1])
    m_neuron_l, v_neuron_l, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                    fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
    
    ### two columns
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ## generate stimuli
    
    ## run network - parameter smaller 
    tc_var_per_stim = dtype(paras[0])
    tc_var_pred = dtype(paras[0])
    [_, _, _, _, alpha_s, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                          v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                          tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                          VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                          v_PE_to_V = v_PE_to_V) 
    
    ## run network - parameter larger 
    tc_var_per_stim = dtype(paras[1])
    tc_var_pred = dtype(paras[1])
    [_, _, _, _, alpha_l, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                          v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                          tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                          VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                          v_PE_to_V = v_PE_to_V) 
    

    ## save
    with open(file_for_data,'wb') as f:
        pickle.dump([paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
                     m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l],f)


   
# %% Robustness: wPE2P

run_cell = False

if run_cell:
    
    ### filename for data
    file_for_data = '../results/data/moments/data_change_wPE2P.pickle'
    
    ## parameters tested
    paras = [0.5, 1.5]
    
    ### one column
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
    
    ## run network - parameter smaller 
    w_PE_to_P_mod = paras[0] * w_PE_to_P
    m_neuron_s, v_neuron_s, _ = run_mfn_circuit(w_PE_to_P_mod, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
    
    ## run network - parameter larger
    w_PE_to_P_mod = paras[1] * w_PE_to_P
    m_neuron_l, v_neuron_l, _ = run_mfn_circuit(w_PE_to_P_mod, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                    fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)

    ### two columns
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)

        
    ## run network - parameter smaller 
    w_PE_to_P_mod = paras[0] * dtype(w_PE_to_P)
    v_PE_to_P_mod = paras[0] * dtype(v_PE_to_P)
    [_, _, _, _, alpha_s, _, _] = run_mfn_circuit_coupled(w_PE_to_P_mod, w_P_to_PE, w_PE_to_PE, v_PE_to_P_mod, 
                                                          v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                          tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                          VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                          v_PE_to_V = v_PE_to_V) 
    
    ## run network - parameter larger 
    w_PE_to_P_mod = paras[1] * dtype(w_PE_to_P)
    v_PE_to_P_mod = paras[1] * dtype(v_PE_to_P)
    [_, _, _, _, alpha_l, _, _] = run_mfn_circuit_coupled(w_PE_to_P_mod, w_P_to_PE, w_PE_to_PE, v_PE_to_P_mod, 
                                                          v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                          tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                          VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                          v_PE_to_V = v_PE_to_V) 

    ## save
    with open(file_for_data,'wb') as f:
        pickle.dump([paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
                     m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l],f)


# %% Robustness: Scale all outcoming weights from M 

run_cell = True

if run_cell:
    
    ### filename for data
    file_for_data = '../results/data/moments/data_change_wP2PE.pickle'
        
    ## parameters tested
    paras = [0.5, 1.5]
    
    ### one column
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
    
    ## run network - parameter smaller 
    w_P_to_PE_mod = paras[0] * w_P_to_PE
    m_neuron_s, v_neuron_s, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE_mod, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)
    
    ## run network - parameter larger
    w_P_to_PE_mod = paras[1] * w_P_to_PE
    m_neuron_l, v_neuron_l, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE_mod, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                    fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V)

    ### two columns
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    ## run network - parameter smaller 
    w_P_to_PE_mod = paras[0] * dtype(w_P_to_PE)
    v_P_to_PE_mod = paras[0] * dtype(v_P_to_PE)
    w_lower2higher = paras[0]
    [_, _, _, _, alpha_s, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE_mod, w_PE_to_PE, v_PE_to_P, 
                                                          v_P_to_PE_mod, v_PE_to_PE, tc_var_per_stim, 
                                                          tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                          VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                          v_PE_to_V = v_PE_to_V, w_lower2higher = w_lower2higher) 
    
    ## run network - parameter larger 
    w_P_to_PE_mod = paras[1] * dtype(w_P_to_PE)
    v_P_to_PE_mod = paras[1] * dtype(v_P_to_PE)
    w_lower2higher = paras[1]
    [_, _, _, _, alpha_l, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE_mod, w_PE_to_PE, v_PE_to_P, 
                                                          v_P_to_PE_mod, v_PE_to_PE, tc_var_per_stim, 
                                                          tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                          VS = VS, VV = VV, w_PE_to_V = w_PE_to_V, 
                                                          v_PE_to_V = v_PE_to_V, w_lower2higher = w_lower2higher) 

    ## save
    with open(file_for_data,'wb') as f:
        pickle.dump([paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
                     m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l],f)
    

    
# %% Robustness: wPE2V

run_cell = False

if run_cell:
    
    ### filename for data
    file_for_data = '../results/data/moments/data_change_wPE2V.pickle'
        
    ## parameters tested
    paras = [0.5, 1.5]
    
    ### one column
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag)
    
    
    ## run network - parameter smaller 
    w_PE_to_V_mod = paras[0] * dtype(w_PE_to_V)
    m_neuron_s, v_neuron_s, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V_mod)
    
    ## run network - parameter larger
    w_PE_to_V_mod = paras[1] * dtype(w_PE_to_V)
    m_neuron_l, v_neuron_l, _ = run_mfn_circuit(w_PE_to_P, w_P_to_PE, w_PE_to_PE, tc_var_per_stim, tau_pe, 
                                                    fixed_input, stimuli_1, VS=VS, VV=VV, w_PE_to_V = w_PE_to_V_mod)
    
    ### two columns
    [w_PE_to_P, w_P_to_PE, w_PE_to_PE, w_PE_to_V, 
     v_PE_to_P, v_P_to_PE, v_PE_to_PE, v_PE_to_V, 
     tc_var_per_stim, tc_var_pred, tau_pe, fixed_input] = default_para_mfn(mfn_flag, one_column=False)
    
    
    ## run network - parameter smaller 
    w_PE_to_V_mod = paras[0] * dtype(w_PE_to_V)
    v_PE_to_V_mod = paras[0] * dtype(v_PE_to_V)
    [_, _, _, _, alpha_s, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                          v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                          tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                          VS = VS, VV = VV, w_PE_to_V = w_PE_to_V_mod, 
                                                          v_PE_to_V = v_PE_to_V_mod) 
    
    ## run network - parameter larger 
    w_PE_to_V_mod = paras[1] * dtype(w_PE_to_V)
    v_PE_to_V_mod = paras[1] * dtype(v_PE_to_V)
    [_, _, _, _, alpha_l, _, _] = run_mfn_circuit_coupled(w_PE_to_P, w_P_to_PE, w_PE_to_PE, v_PE_to_P, 
                                                          v_P_to_PE, v_PE_to_PE, tc_var_per_stim, 
                                                          tc_var_pred, tau_pe, fixed_input, stimuli_2, 
                                                          VS = VS, VV = VV, w_PE_to_V = w_PE_to_V_mod, 
                                                          v_PE_to_V = v_PE_to_V_mod) 

    ## save
    with open(file_for_data,'wb') as f:
        pickle.dump([paras, stimuli_1, trial_duration_1, n_trials, trial_duration_2, stimuli_2, alpha_s, alpha_l,
                     m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l],f)


# %% Robustness: additional top-down input

run_cell = False

if run_cell:
    
    ### choose mean-field network to simulate
    mfn_flag = '10' # valid options are '10', '01', '11
    
    ### filename for data
    file_for_data = '../results/data/moments/data_add_top_down.pickle'
    file_temp = '../results/data/moments/data_temp.pickle'
    
    ## parameters tested
    paras = [0.5, 1]
    add_input = np.zeros(8)
    
    ### one column
    
    ### run network - add top-down 1
    add_input[2:4] = paras[0]
    add_input[5] = paras[0]
    add_input[7] = paras[0]
    [_, _, _, _, _, m_neuron_s, v_neuron_s] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_temp, add_input = add_input,
                                                                          trial_duration = np.int32(150000), 
                                                                          num_values_per_trial = np.int32(300))
    
    ## run network - add top-down 2
    add_input[2:4] = paras[1]
    add_input[5] = paras[1]
    add_input[7] = paras[1]
    [_, _, _, _, _, m_neuron_l, v_neuron_l] = simulate_example_pe_circuit(mfn_flag, 5, 2, file_temp, add_input = add_input,
                                                                          trial_duration = np.int32(150000), 
                                                                          num_values_per_trial = np.int32(300))
    
    ### two columns
    
    ## run network - add top-down 1
    cell_ids = np.array([2,3,5,7])
    
    _, alpha_s = simulate_effect_baseline(mfn_flag, 5, 5, 0, [paras[0]], cell_ids, record_last_alpha = True, n_trials = np.int32(150))
    
    ## run network - ad top-down 2
    _, alpha_l = simulate_effect_baseline(mfn_flag, 5, 5, 0, [paras[1]], cell_ids, record_last_alpha = True, n_trials = np.int32(150))
    
    ## save
    with open(file_for_data,'wb') as f:
        pickle.dump([paras, alpha_s, alpha_l, m_neuron_s, m_neuron_l, v_neuron_s, v_neuron_l],f)
    
   