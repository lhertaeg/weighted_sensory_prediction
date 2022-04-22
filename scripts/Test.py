#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:34:36 2022

@author: loreen.hertaeg
"""

import numpy as np
import matplotlib.pyplot as plt

# %% STDP to get necessary connectivity between PE neurons and perfect integrator?

flag = 0

if flag==1:
    
    w, eta = 1, 1e-2
    A, B = -1, 1 # Hebbian and Anti-Hebbian depending on sign of A and B
    tau_A, tau_B = 1, 1
    T = 0 # sometimes necessary (kind of normalisation or delay)
    
    S_arr = np.random.uniform(1,5,size=500)
    P = np.mean(S_arr)
    
    weight = np.zeros_like(S_arr)
    
    for i, S in enumerate(S_arr):
        
        nPE = np.maximum(P-S,0)
        pPE = np.maximum(S-P,0)
        E = P
        #I = nPE - np.random.uniform(0.1,0.3) # to simulate that there is a delay
        I = pPE - np.random.uniform(0.0,0.2)
       
        #r_pre = nPE
        r_pre = pPE
        #r_post = E
        r_post = I
        
        HA = 0.5 * (np.sign(r_pre-r_post + T) + 1)
        HB = 0.5 * (np.sign(r_post-r_pre - T) + 1) 
        w += eta * (A * np.exp(-(r_pre-r_post + T)/tau_A) * HA + B * np.exp(-(r_post-r_pre - T)/tau_B) * HB)
        
        if w<0:
            w=0
        
        weight[i] = w
        
        
    plt.figure()
    plt.plot(weight)
    
    
    # r_post = 1
    # r_pre_all = np.linspace(0,2,51)
    # dw = np.zeros_like(r_pre_all)
    
    # A, B = 1, -1
    # tau_A, tau_B = 0.2, 0.2
    # T = 0.25
    
    # for i, r_pre in enumerate(r_pre_all):
    
    #     HA = 0.5 * (np.sign(r_pre-r_post + T) + 1)
    #     HB = 0.5 * (np.sign(r_post-r_pre - T) + 1) 
    #     dw[i] = A * np.exp(-(r_pre-r_post + T)/tau_A) * HA + B * np.exp(-(r_post-r_pre - T)/tau_B) * HB
        
    
    # plt.figure()
    # plt.plot(r_pre_all-r_post, dw)
    # plt.axhline(0, color='r', ls=':')
    # plt.axvline(0, color='r', ls=':')


# %% All together

# Think about the necessary constraints for the parameters (min, max, relations)
# Here, for instance, it seems that to cover trail mean and variance properly, tau's need to be in a sensible relation

flag = 0

if flag==1:
    
    N_sample = 1000
    N_trials = 1000
    S = np.random.uniform(0, 10, size = N_trials)
    std_per_trial = np.sqrt(np.random.uniform(2, 8, size = N_trials)) # std per trail
    
    # parameters for mean_P-circuit
    fac = 3e5 # fac = weight/tau
    P = np.mean(S) # 0
    
    # parameters for var_P-circuit
    P_s2 = 0
    fac_2 = 3e5
    
    # parameters for current trial mean
    SP = 0
    tau_SP = 100
    
    # parameters for current (trial) variance
    SP_s2 = 0
    tau_Ss2 = 300

    P_vs_S = np.zeros_like(S)
    S2_vs_S = np.zeros_like(S)
    P2_vs_S = np.zeros_like(S)
    
    for i, s in enumerate(S):
        
        st = np.random.normal(loc=s, scale=std_per_trial[i], size = N_sample) 
        std_per_trial[i] = np.std(st)
        
        for t in np.arange(N_sample):
            
            # trial-related (operating on shorter time scales)
            SP = (1-1/tau_SP) * SP + st[t]/tau_SP
            pPE_mom = (np.maximum(st[t] - SP,0))**2
            nPE_mom = (SP - np.maximum(st[t],0))**2
            
            SP_s2 = (1-1/tau_Ss2) * SP_s2 + (nPE_mom + pPE_mom)/tau_Ss2
            
            # overall (operating on longer time scales)
            pPE_mean = (np.maximum(st[t] - P,0))**2
            nPE_mean = (np.maximum(P - st[t],0))**2
            P += (pPE_mean - nPE_mean)/fac
            
            pPE_SD = (np.maximum(SP_s2 - P_s2,0))**2
            nPE_SD = (np.maximum(P_s2 - SP_s2,0))**2
            P_s2 += (pPE_SD - nPE_SD)/fac_2
                
        P_vs_S[i] = P
        S2_vs_S[i] = SP_s2
        P2_vs_S[i] = P_s2
    
    plt.figure()
    plt.plot(S2_vs_S,'m', alpha=0.5)
    plt.plot(std_per_trial**2,'k.',alpha=0.5)
    plt.plot(P_vs_S,'b')
    plt.plot(P2_vs_S,'r')
    plt.ylim(bottom=0)


# %% Test variance over trails - I

# Different types of PE neurons and memory neurons:
    # PE neurons for mean & memory neuron holding the mean
    # PE neurons for varaince & memory neurons holding the variance

flag = 1

if flag==1:
    
    S1 = np.random.uniform(0, 10, size=1000)
    S2 = np.random.uniform(3, 7, size=1000)
    S = np.concatenate((S1,S2))

    #S = np.random.uniform(0, 10, size=2000)
    
    eta = 1e-4
    tau = 100
    tau_SD = 10
    E = 0
    SD = 0
    SD_long = 0
    E_SD = 0
    
    rate_1 = np.zeros_like(S)
    rate_2 = np.zeros_like(S)
    rate_3 = np.zeros_like(S)
    
    for i, s in enumerate(S):
        
        pPE = (np.maximum(s - E,0))**2
        nPE = (np.maximum(E - s,0))**2
        E_SD = (1-1/tau) * E_SD + s/tau
        SD = (1-1/tau_SD) * SD + (nPE+pPE)/tau_SD
        
        for t in np.arange(500):
            
            pPE_mean = (np.maximum(s - E,0))**2
            nPE_mean = (np.maximum(E - s,0))**2
            E += eta * (pPE_mean - nPE_mean)/tau
            
            pPE_SD = (np.maximum(SD - SD_long,0))**2
            nPE_SD = (np.maximum(SD_long - SD,0))**2
            SD_long += eta * (pPE_SD - nPE_SD)/tau
                  
        rate_1[i] = E
        rate_2[i] = SD
        rate_3[i] = SD_long
    
    plt.figure()
    plt.plot(rate_1,'b')
    plt.plot(rate_2,'m')
    plt.plot(rate_3,'r')
    

# %% Test "current" varaince - II

flag = 0

if flag==1:

    S1 = np.random.uniform(0, 10, size=1000)
    S2 = np.random.uniform(3, 7, size=1000)
    S = np.concatenate((S1,S2))
    
    tau = 100
    tau_SD = 10
    E = 0
    SD = 0
    
    rate = np.zeros_like(S)
    
    for i, s in enumerate(S):
        
        #for t in np.arange(1000):
            
        pPE = (np.maximum(s - E,0))**2
        nPE = (np.maximum(E - s,0))**2
        E = (1-1/tau) * E + s/tau
        SD = (1-1/tau_SD) * SD + (nPE+pPE)/tau_SD
            
        rate[i] = SD
        
    plt.figure()
    plt.plot(rate)

# %% Test "current" varaince - I

# simply the current squared activity of the PE neurons ...

flag = 0

if flag==1:

    S = np.random.uniform(0, 10, size=2000)
    
    eta = 1e-4
    tau = 100
    E = 0
    
    rate = np.zeros_like(S)
    
    for i, s in enumerate(S):
        
        E = (1-1/tau) * E + s/tau   
        pPE = (np.maximum(s - E,0))**2
        nPE = (np.maximum(E - s,0))**2
            
        rate[i] = nPE + pPE
        
    plt.figure()
    plt.plot(rate)

# %% Test mean over trails - III

flag = 0

if flag==1:

    S = np.random.uniform(0, 10, size=2000)
    
    eta = 1e-4
    tau = 100
    E = 0
    
    rate = np.zeros_like(S)
    
    for i, s in enumerate(S):
        
        for t in np.arange(1000):
            
            pPE = (np.maximum(s - E,0))**2
            nPE = (np.maximum(E - s,0))**2
            E += eta * (pPE - nPE)/tau
            
        rate[i] = E
        
    
    plt.figure()
    plt.plot(rate)

# %% Test mean over trails - II

flag = 1

if flag==1:

    S = np.random.uniform(0, 10, size=2000)
    
    eta = 1e-4
    tau = 10
    E = 0
    
    rate = np.zeros_like(S)
    
    for i, s in enumerate(S):
        
        for t in np.arange(1000):
            
            PE = (s - E)
            E += eta * PE/tau
            
        rate[i] = E
        
    
    plt.figure()
    plt.plot(rate)


# %% Test mean over trails - I

flag = 0

if flag==1:

    S = np.random.uniform(0, 10, size=2000)
    
    tau = 100
    E = 0
    
    rate = np.zeros_like(S)
    
    for i, s in enumerate(S):
        
        #for t in np.arange(1000):
            
        E = (1-1/tau) * E + s/tau
            
        rate[i] = E
        
    
    plt.figure()
    plt.plot(rate)
    
# %% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# %% Test filtered sensory input as S or P ...

# Main differences are smoothness of results (robustness) and by using a line attractor the memory effect
# One could dig deeper, so there might be subtle differences between encoding mean of trial vs. mean of samples but I don't think it is worth it

flag = 0

if flag==1:
    
    N_sample = 500 # 100 - 1000
    N_trials = 1000
    S = np.random.uniform(0, 10, size = N_trials)
    std_per_trial = np.sqrt(np.random.uniform(2, 8, size = N_trials)) # std per trail
    
    # parameters
    Es = 0
    P_2 = 0
    
    tau_P2 = 10000
    tau_Es = 10000

    # Main
    P = np.zeros((len(S),2))
    nPE = np.zeros((len(S),2))
    pPE = np.zeros((len(S),2))
    
    for i, s in enumerate(S):
        
        st = np.random.normal(loc=s, scale=std_per_trial[i], size = N_sample) 
        std_per_trial[i] = np.std(st)
        
        for t in np.arange(N_sample):
            
            # Sensory input
            Es = (1-1/tau_Es) * Es + st[t]/tau_Es
            
            # PE neurons in two different scenarios
            pPE_1 = (np.maximum(st[t] - Es,0))**2
            nPE_1 = (np.maximum(Es - st[t],0))**2
            
            pPE_2 = (np.maximum(Es - P_2,0))**2
            nPE_2 = (np.maximum(P_2 - Es,0))**2
            
            # Prediction in both scenarios
            P_1 = Es
            P_2 += (pPE_2 - nPE_2)/tau_P2
            
                
        P[i,0] = P_1
        P[i,1] = P_2
        nPE[i,0] = nPE_1
        nPE[i,1] = nPE_2
        pPE[i,0] = pPE_1
        pPE[i,1] = pPE_2
    
    plt.figure()
    plt.plot(P[:,0],'r')
    plt.plot(P[:,1],'b')
    
    plt.figure()
    plt.plot(nPE[:,0],'r')
    plt.plot(nPE[:,1],'b')
    
    plt.figure()
    plt.plot(pPE[:,0],'r')
    plt.plot(pPE[:,1],'b')