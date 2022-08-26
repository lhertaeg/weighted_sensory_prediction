# %% Import

import numpy as np
import matplotlib.pyplot as plt

# %% Kalman filter for polynom 3rd degree

# This source code is public domain
# Autor: Christian Schirm

flag = 0

if flag==1:

    # Generate polynomial
    nSteps = 301
    coeff = [-50, 70, -16, 1]
    sigmaNoise = 50
    sigmaPrior = 100
    xMax = 10
    
    ts = np.linspace(0,xMax,nSteps)
    deltaT = ts[1] - ts[0]
    nPoly = len(coeff)
    A = np.array([ts**i for i in range(nPoly)])
    y_polynomial = coeff @ A
    
    # Noise
    np.random.seed(1)
    noise = sigmaNoise*np.random.randn(nSteps)
    
    # Add noise to the signal
    y = y_polynomial + noise
    
    # Prepare Kalman estimation
    D = np.zeros((nPoly,nPoly))
    D[(np.arange(nPoly-1), np.arange(nPoly-1)+1)] = 1
    Dt = D*deltaT
    F = np.identity(nPoly) + Dt + Dt @ Dt/2 + Dt @ Dt @ Dt/6
    H = np.zeros((1,nPoly))
    H[0,0] = 1
    
    # Initialize Kalman estimation
    x = np.zeros(nPoly)
    P = sigmaPrior**2 * np.identity(nPoly)
    # components = A / nSteps
    # P = sigmaPrior**2 * np.linalg.inv(components @ components.T)  # Constant variance prior model
    
    # Start Kalman iteration
    yEst = []
    ySigma = []
    for i in range(len(y)):
        # Propagate (actually "prediction")
        if i > 0:
            x = F @ x
            P = F @ P @ F.T
    
        # Estimate (actually "correction")
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + sigmaNoise**2)
        x = x + K @ (y[i] - H @ x)
        P = (np.identity(nPoly) - K @ H) @ P
        ySigma.append(P[0,0])
        yEst.append(x[0])
    
    ySigma = np.sqrt(ySigma)
    
    # Plot
    plt.figure(figsize=(5,3.5))
    #plt.fill_between(ts,y_polynomial-ySigma, y_polynomial+ySigma, color='0.2', alpha=0.17, label='Konfidenzintervall', lw=0)
    plt.fill_between(ts,yEst-ySigma, yEst+ySigma, color='0.2', alpha=0.17, label='Konfidenzintervall', lw=0)
    plt.plot(ts,y,'.-', color='C1', markersize=4, linewidth=0.4, alpha=0.6, label='Polynom + Rauschen')
    plt.plot(ts,y_polynomial,'C3-', label='Polynom 3. Grades')
    plt.plot(ts,yEst,'C0-',  label='Kalman-Schätzung')
    plt.xlabel('Zeit')
    plt.legend(loc=4)
    plt.tight_layout()
    #plt.savefig('Kalman_Polynom_Test.svg')
    #plt.savefig('Kalman_Polynom_Test.png')
    #plt.show()


# %% Kalman filter for random variable drawn from uniform distribution

# based on implementation above

flag = 0

if flag==1:

    # Generate polynomial
    nSteps = 301
    sigmaNoise = 1
    sigmaPrior = 1
    xMax = 10
    nPoly = 1 #4
    
    ts = np.linspace(0,xMax,nSteps)
    deltaT = ts[1] - ts[0]
    y = np.random.uniform(1,5,size=nSteps)
    
    # Prepare Kalman estimation
    D = np.zeros((nPoly,nPoly))
    D[(np.arange(nPoly-1), np.arange(nPoly-1)+1)] = 1
    Dt = D*deltaT
    F = np.identity(nPoly) #+ Dt + Dt @ Dt/2 + Dt @ Dt @ Dt/6
    H = np.zeros((1,nPoly))
    H[0,0] = 1
    
    # Initialize Kalman estimation
    x = np.zeros(nPoly)
    P = sigmaPrior**2 * np.identity(nPoly)
    # components = A / nSteps
    # P = sigmaPrior**2 * np.linalg.inv(components @ components.T)  # Constant variance prior model
    
    # Start Kalman iteration
    yEst = []
    ySigma = []
    for i in range(len(y)):
        # Propagate (actually "prediction")
        if i > 0:
            x = F @ x
            P = F @ P @ F.T
    
        # Estimate (actually "correction")
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + sigmaNoise**2)
        x = x + K @ (y[i] - H @ x)
        P = (np.identity(nPoly) - K @ H) @ P
        ySigma.append(P[0,0])
        yEst.append(x[0])
    
    ySigma = np.sqrt(ySigma)
    
    # Plot
    plt.figure(figsize=(5,3.5))
    #plt.fill_between(ts,y-ySigma, y+ySigma, color='0.2', alpha=0.17, label='Konfidenzintervall', lw=0)
    plt.fill_between(ts,yEst-ySigma, yEst+ySigma, color='0.2', alpha=0.17, label='Konfidenzintervall', lw=0)
    plt.plot(ts,y,'.-', color='C1', markersize=4, linewidth=0.4, alpha=0.6, label='data')
    plt.plot(ts,yEst,'C0-',  label='Kalman-Schätzung')
    plt.xlabel('Zeit')
    plt.legend(loc=4)
    plt.tight_layout()
    
    
# %% Kalman filter for random variable drawn from uniform distribution (simplified)

# based on implementation above

flag = 0

if flag==1:

    # Generate polynomial
    nSteps = 301
    sigmaNoise = 1
    sigmaPrior = 1
    xMax = 10
    
    ts = np.linspace(0,xMax,nSteps)
    deltaT = ts[1] - ts[0]
    
    #np.random.seed(1)
    y = np.random.uniform(1,5,size=nSteps)
    
    # Prepare Kalman estimation
    F = 1
    H = 1
    
    # Initialize Kalman estimation
    x = 0
    P = sigmaPrior**2
    
    # Start Kalman iteration
    yEst = []
    ySigma = []
    for i in range(len(y)):
        # Propagate (actually "prediction")
        if i > 0:
            x = F * x
            P = F * P * F
    
        # Estimate (actually "correction")
        K = P * H / (H * P * H + sigmaNoise**2)
        x = x + K * (y[i] - H * x)
        P = (1 - K * H) * P
        ySigma.append(P)
        yEst.append(x)
    
    ySigma = np.sqrt(ySigma)
    
    # Plot
    plt.figure(figsize=(5,3.5))
    #plt.fill_between(ts,y-ySigma, y+ySigma, color='0.2', alpha=0.17, label='Konfidenzintervall', lw=0)
    plt.fill_between(ts,yEst-ySigma, yEst+ySigma, color='0.2', alpha=0.17, label='Konfidenzintervall', lw=0)
    plt.plot(ts,y,'.-', color='C1', markersize=4, linewidth=0.4, alpha=0.6, label='data')
    plt.plot(ts,yEst,'C0-',  label='Kalman-Schätzung')
    plt.xlabel('Zeit')
    plt.legend(loc=4)
    plt.tight_layout()
    
    
# %% Kalman filter for random variable drawn from two uniform distribution (simplified)

# based on implementation above

flag = 1

if flag==1:

    # sensory data
    nSteps = 301
    sigmaNoise = 1   # 1e-5
    sigmaPrior = 1
    
    #np.random.seed(1)
    y1 = np.random.uniform(1,5,size=nSteps)
    y2 = np.random.uniform(7,7,size=nSteps)
    y = np.concatenate((y1,y2))
    
    deltaT = 0.01 #ts[1] - ts[0]
    ts = np.arange(0, 2*nSteps*deltaT, deltaT)
    
    # Prepare Kalman estimation
    F = 1
    H = 1
    
    # Initialize Kalman estimation
    x = 0
    P = sigmaPrior**2
    
    # Start Kalman iteration
    yEst = []
    ySigma = []
    for i in range(len(y)):
        # Propagate (actually "prediction")
        if i > 0:
            x = x #F * x
            P = P #F * P * F
    
        # Estimate (actually "correction")
        K = P / (P + sigmaNoise**2) #P * H / (H * P * H + sigmaNoise**2)
        x = x + K * (y[i] - x) # x + K * (y[i] - H * x)
        P = (1 - K) * P # (1 - K * H) * P
        ySigma.append(P)
        yEst.append(x)
    
    ySigma = np.sqrt(ySigma)
    
    # Plot
    plt.figure(figsize=(5,3.5))
    #plt.fill_between(ts,y-ySigma, y+ySigma, color='0.2', alpha=0.17, label='Konfidenzintervall', lw=0)
    plt.fill_between(ts,yEst-ySigma, yEst+ySigma, color='0.2', alpha=0.17, label='Konfidenzintervall', lw=0)
    plt.plot(ts,y,'.-', color='C1', markersize=4, linewidth=0.4, alpha=0.6, label='data')
    plt.plot(ts,yEst,'C0-',  label='Kalman-Schätzung')
    plt.xlabel('Zeit')
    plt.legend(loc=4)
    plt.tight_layout()