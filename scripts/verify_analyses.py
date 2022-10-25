#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:34:41 2022

@author: loreen.hertaeg
"""

# %% Import

import numpy as np
import matplotlib.pyplot as plt

# %% functions

def run_trial(stims, n, l, dt, tau):
    
    y = np.zeros(n)

    x = np.zeros(l*n)
    s = np.repeat(stims, l)

    for i in range(l*n):
        
        x[i] = (1-dt/tau) * x[i-1] + s[i] * dt/tau
    
    
    for j in range(n):
        
        i = 0
        
        while i <= j:
            
            y[j] += stims[i] * np.exp(-l/tau)**((j+1)-(i+1))
            i += 1
            
        y[j] *= (1 - np.exp(-l/tau))
        
    return s, x, y
    

def run_variance_trial(stims, n, l, dt, tau, tauv):
    
    y = np.zeros(n)
    w = np.zeros(n)
    x = np.zeros(l*n)
    v = np.zeros(l*n)
    s = np.repeat(stims, l)
    
    k = tau/(tau-2*tauv)
    h = np.exp(-2*l/tau) - np.exp(-l/tauv)

    for i in range(l*n):
        
        x[i] = (1-dt/tau) * x[i-1] + s[i] * dt/tau
        v[i] = (1-dt/tauv) * v[i-1] + (s[i]-x[i])**2 * dt/tauv
    
    
    for j in range(n):
        
        i = 0
        
        while i <= j:
            
            y[j] += (stims[i] * np.exp(-l/tau)**(j-i)) * (1 - np.exp(-l/tau))
            w[j] += k*h * (s[i] - y[i-1])**2 * np.exp(-l/tauv)**(j-i)
            i += 1
        
    return s, x, v, y, w
    

# %% test rM for a trial

flag = 0

if flag==1:

    dt = 1
    tau = 500
    l = 500 #15
    n = 10
    
    ### one example
    stims = np.random.uniform(1, 5, size=n)
    s, x, y = run_trial(stims, n, l, dt, tau)
        
    time_end = np.arange(1,n+1) * l
    
    plt.figure()
    plt.plot(np.arange(len(x)),x)
    plt.plot(time_end, y, 'or')
    plt.plot(time_end-10, stims, 'ob')
    
    ### average over several examples
    n_repeats = 100
    x_n = np.zeros(n_repeats)
    y_n = np.zeros(n_repeats)
    
    for k in range(n_repeats):
    
        stims = np.random.uniform(1,5,size=n)
        s, x, y = run_trial(stims, n, l, dt, tau)
        
        x_n[k] = x[-1]
        y_n[k] = y[-1]
    
    mean_stims = np.mean(np.random.uniform(1,5,1000))
    z = mean_stims * (1 - np.exp(-n*l/tau))
        
    print(np.mean(x_n))
    print(np.mean(y_n))
    print(z)


# %% test rM over several trials

flag = 0

if flag==1:
    
    dt = 1
    tau = 1000
    
    l = 500 # stimulus duration
    n = 10 # number of stimuli per trial
    
    N = 50 # number of trials
    L = n*l # trial duration
    
    mu_across_trials = 10
    sig_across_trials = 3
    sig_within_trials = 2
    
    stims = np.array([])
    
    ### create stimulus
    for i in range(N):
        
        mu_trial = np.random.normal(mu_across_trials, sig_across_trials)
        stims = np.append(stims, np.random.normal(mu_trial, sig_within_trials, size = n))
    
    ### run model
    s, x, y = run_trial(stims, n*N, l, dt, tau)
    
    ### plot
    plt.figure()
    plt.plot(np.arange(len(x)),x)
    
    time_end = np.arange(1,(n*N)+1) * l
    plt.plot(time_end, y, 'or')
    
    ### mean across trials
    x_theory_end = mu_across_trials * (1 - np.exp(-N*L/tau))
    x_end = np.mean(np.array_split(x,N),0)[-1]
    print(x_end)
    print(x_theory_end)
    

# %% test rV for a trial

flag = 1

if flag==1:

    dt = 1
    tau = 500
    tauv = 1000
    l = 500 #15
    n = 10
    
    ### one example
    stims = np.random.uniform(1, 5, size=n)
    s, x, v, y, w = run_variance_trial(stims, n, l, dt, tau, tauv)
        
    time_end = np.arange(1,n+1) * l
    
    plt.figure()
    plt.plot(np.arange(len(x)),x)
    plt.plot(time_end, y, 'or')
    #plt.plot(np.arange(len(x)),s)
    
    plt.figure()
    plt.plot(np.arange(len(v)),v)
    plt.plot(time_end, w, 'or')
    
    test = np.zeros(l)
    for j in range(l):
        
        test[j] = 0
        i=0
        
        while i<=j:
            test[j] += np.exp(-(l-i)/tauv) * (x[i]-s[i])**2 / tauv
            i += 1
    
    plt.plot(test)
    
    (s[0] - 0)**2 * tau/(tau - 2*tauv) * (np.exp(-2*l/tau) - np.exp(-l/tauv))
    
    # ok, it seems that the estimate for variance at first stimulus (end) is correct but it then diverges from reality
    # that means that I have to check the equations again, to be precise, check (A) for subsequent stimuli, 
    # (B) for second, third and so on ...
    
    # already "test" for first does not seem to perfectly match (the dicrepancy gets bigger with longer trial duration)
    # maybe start there ... 
    
    
    
    
    


    ### average over several examples
    # n_repeats = 100
    # x_n = np.zeros(n_repeats)
    # y_n = np.zeros(n_repeats)
    
    # for k in range(n_repeats):
    
    #     stims = np.random.uniform(1,5,size=n)
    #     s, x, v, y, w = run_variance_trial(stims, n, l, dt, tau, tauv)
        
    #     x_n[k] = x[-1]
    #     y_n[k] = y[-1]
    
    # mean_stims = np.mean(np.random.uniform(1,5,1000))
    # z = mean_stims * (1 - np.exp(-n*l/tau))
        
    # print(np.mean(x_n))
    # print(np.mean(y_n))
    # print(z)