import numpy as np
import linAlg as linAlg
import helper as helper


def makeMagneticFieldPrediction(predictionsLocations, modelParameters):
    ''' Make magnetic field prediction
    Input is a 3xN matrix of positions '''

    if predictionsLocations.shape == 1:
        print("Error: Input should be a 3xN matrix")
    elif predictionsLocations.shape[0] != 3: 
        print("Error: Input should be a 3xN matrix")
    nablaPhiLinPredictions = linAlg.matrix3DTo2DVertical(nablaPhiLin3D(predictionsLocations, modelParameters))
    f = nablaPhiLinPredictions @ modelParameters['GPweights']
    return f.reshape(-1, 3).T

'''Hilbert space reduced rank'''
def setNumberOfBasisFunctions(Nm, modelParameters):
    theta = modelParameters['theta']
    modelParameters["Nm"] = Nm 
    modelParameters["NmLinCf"] = Nm + 3
    modelParameters["indices"], modelParameters["Lambda"] = Lambda3D(modelParameters)
    modelParameters["lambdaLinCf"] = linAlg.blockDiag(np.eye(3)*theta[3], modelParameters["Lambda"])

    return

def Phi3D(X, modelParameters):
    ''' Copyright (C) 2022 by Frida Viset
        Adapted into python by Thomas Edridge '''
    Nm = modelParameters["Nm"]
    indices = modelParameters["indices"]     
    domain = modelParameters["domain"]
    xLower = domain[0, 0]
    xUpper = domain[0, 1]
    yLower = domain[1, 0]
    yUpper = domain[1, 1]
    zLower = domain[2, 0]
    zUpper = domain[2, 1]
    
    # Basis functions for the SE kernel function
    # Call this function first to find indices: [Indices, Lambda]=Lambda2D(m);
    j1 = indices[:, 0]
    j2 = indices[:, 1]
    j3 = indices[:, 2]

    N = X.shape[1]
    phi = np.zeros((N, Nm))
    
    for i in range(N):
        phi[i, :] = (
            1 / np.sqrt(0.5 * (xUpper - xLower)) * np.sin(np.pi * j1 * (X[0, i] - xLower) / (xUpper - xLower)) *
            1 / np.sqrt(0.5 * (yUpper - yLower)) * np.sin(np.pi * j2 * (X[1, i] - yLower) / (yUpper - yLower)) *
            1 / np.sqrt(0.5 * (zUpper - zLower)) * np.sin(np.pi * j3 * (X[2, i] - zLower) / (zUpper - zLower))
        )

    return phi

def Lambda3D(modelParameters):
    ''' Copyright (C) 2022 by Frida Viset
        Adapted into python by Thomas Edridge '''
    Nm = modelParameters["Nm"]
    domain = modelParameters["domain"]
    xLower = domain[0, 0]
    xUpper = domain[0, 1]
    yLower = domain[1, 0]
    yUpper = domain[1, 1]
    zLower = domain[2, 0]
    zUpper = domain[2, 1]
    
    
    # Eigenvalues for the 3D squared exponential basis functions
    maxTestValues = 20
    indices = []

    # Generate indices for the basis functions
    for i in range(1, maxTestValues + 1):
        for j in range(1, maxTestValues + 1):
            for k in range(1, maxTestValues + 1):
                indices.append([i, j, k])
    
    indices = np.array(indices)
    
    j1 = indices[:, 0]
    j2 = indices[:, 1]
    j3 = indices[:, 2]

    n = indices.shape[0]
    eigenValuesCandidates = np.zeros(n)

    for j in range(n):
        eigenValue = (np.pi * j1[j] / (xUpper - xLower)) ** 2 + (np.pi * j2[j] / (yUpper - yLower)) ** 2 + (np.pi * j3[j] / (zUpper - zLower)) ** 2
        eigenValuesCandidates[j] = spectralSE(np.sqrt(eigenValue), modelParameters)

    # Sort eigenvalues in descending order
    eigenValuesSorted = np.sort(eigenValuesCandidates)[::-1]
    eigenValuesIndicesSorted = np.argsort(eigenValuesCandidates)[::-1]
    
    indices = indices[eigenValuesIndicesSorted][:Nm]

    # Create the Lambda matrix
    Lambda = np.eye(Nm)
    for j in range(Nm):
        Lambda[j, j] = eigenValuesSorted[j]

    return indices, Lambda

def nablaPhi3D(X, modelParameters):
    ''' Copyright (C) 2022 by Frida Viset
    Adapted into python by Thomas Edridge '''
    Nm = modelParameters['Nm']
    indices = modelParameters['indices']     
    domain = modelParameters["domain"]
    xLower = domain[0, 0]
    xUpper = domain[0, 1]
    yLower = domain[1, 0]
    yUpper = domain[1, 1]
    zLower = domain[2, 0]
    zUpper = domain[2, 1]
    
    # Basis functions for the SE kernel function
    j1 = indices[:, 0]
    j2 = indices[:, 1]
    j3 = indices[:, 2]

    N = X.shape[1]
    nablaPhi = np.zeros((3, Nm, N))

    for i in range(N):
        nablaPhi[0, :, i] = (
            1 / np.sqrt(0.5 * (xUpper - xLower)) * np.cos(np.pi * j1 * (X[0, i] - xLower) / (xUpper - xLower)) * np.pi * j1 / (xUpper - xLower) *
            1 / np.sqrt(0.5 * (yUpper - yLower)) * np.sin(np.pi * j2 * (X[1, i] - yLower) / (yUpper - yLower)) *
            1 / np.sqrt(0.5 * (zUpper - zLower)) * np.sin(np.pi * j3 * (X[2, i] - zLower) / (zUpper - zLower))
        )
        
        nablaPhi[1, :, i] = (
            1 / np.sqrt(0.5 * (xUpper - xLower)) * np.sin(np.pi * j1 * (X[0, i] - xLower) / (xUpper - xLower)) *
            1 / np.sqrt(0.5 * (yUpper - yLower)) * np.cos(np.pi * j2 * (X[1, i] - yLower) / (yUpper - yLower)) * np.pi * j2 / (yUpper - yLower) *
            1 / np.sqrt(0.5 * (zUpper - zLower)) * np.sin(np.pi * j3 * (X[2, i] - zLower) / (zUpper - zLower))
        )
        
        nablaPhi[2, :, i] = (
            1 / np.sqrt(0.5 * (xUpper - xLower)) * np.sin(np.pi * j1 * (X[0, i] - xLower) / (xUpper - xLower)) *
            1 / np.sqrt(0.5 * (yUpper - yLower)) * np.sin(np.pi * j2 * (X[1, i] - yLower) / (yUpper - yLower)) *
            1 / np.sqrt(0.5 * (zUpper - zLower)) * np.cos(np.pi * j3 * (X[2, i] - zLower) / (zUpper - zLower)) * np.pi * j3 / (zUpper - zLower)
        )

    return nablaPhi

def nablaPhiLin3D(X, modelParameters):
    ''' Copyright (C) 2022 by Frida Viset
    Adapted into python by Thomas Edridge '''
    Nm = modelParameters['Nm']
    indices = modelParameters['indices']     
    domain = modelParameters["domain"]
    xLower = domain[0, 0]
    xUpper = domain[0, 1]
    yLower = domain[1, 0]
    yUpper = domain[1, 1]
    zLower = domain[2, 0]
    zUpper = domain[2, 1]
    
    # Basis functions for the SE kernel function
    j1 = indices[:, 0]
    j2 = indices[:, 1]
    j3 = indices[:, 2]

    N = X.shape[1]
    nablaPhiLin = np.zeros((3, Nm+3, N))

    for i in range(N):
        nablaPhiLin[0:3, 0:3, i] = np.eye(3)
        nablaPhiLin[0, 3:, i] = (
            1 / np.sqrt(0.5 * (xUpper - xLower)) * np.cos(np.pi * j1 * (X[0, i] - xLower) / (xUpper - xLower)) * np.pi * j1 / (xUpper - xLower) *
            1 / np.sqrt(0.5 * (yUpper - yLower)) * np.sin(np.pi * j2 * (X[1, i] - yLower) / (yUpper - yLower)) *
            1 / np.sqrt(0.5 * (zUpper - zLower)) * np.sin(np.pi * j3 * (X[2, i] - zLower) / (zUpper - zLower))
        )
        
        nablaPhiLin[1, 3:, i] = (
            1 / np.sqrt(0.5 * (xUpper - xLower)) * np.sin(np.pi * j1 * (X[0, i] - xLower) / (xUpper - xLower)) *
            1 / np.sqrt(0.5 * (yUpper - yLower)) * np.cos(np.pi * j2 * (X[1, i] - yLower) / (yUpper - yLower)) * np.pi * j2 / (yUpper - yLower) *
            1 / np.sqrt(0.5 * (zUpper - zLower)) * np.sin(np.pi * j3 * (X[2, i] - zLower) / (zUpper - zLower))
        )
        
        nablaPhiLin[2, 3:, i] = (
            1 / np.sqrt(0.5 * (xUpper - xLower)) * np.sin(np.pi * j1 * (X[0, i] - xLower) / (xUpper - xLower)) *
            1 / np.sqrt(0.5 * (yUpper - yLower)) * np.sin(np.pi * j2 * (X[1, i] - yLower) / (yUpper - yLower)) *
            1 / np.sqrt(0.5 * (zUpper - zLower)) * np.cos(np.pi * j3 * (X[2, i] - zLower) / (zUpper - zLower)) * np.pi * j3 / (zUpper - zLower)
        )
    return nablaPhiLin

def jacobianPhi3D(X, modelParameters):
    ''' Copyright (C) 2022 by Frida Viset
    Adapted into python by Thomas Edridge '''
    Nm = modelParameters['Nm']
    indices = modelParameters['indices']     
    domain = modelParameters["domain"]
    xLower = domain[0, 0]
    xUpper = domain[0, 1]
    yLower = domain[1, 0]
    yUpper = domain[1, 1]
    zLower = domain[2, 0]
    zUpper = domain[2, 1]
    
    # Basis functions for the SE kernel function
    j = indices

    N = X.shape[1]
    J = np.zeros((3, 3, Nm, N))

    a = np.array([xLower, yLower, zLower])
    b = np.array([xUpper, yUpper, zUpper])

    f = np.zeros((Nm, 3))
    for d in range(3):
        f[:, d] = (np.pi * j[:, d]) / (b[d] - a[d])

    for i in range(N):
        s = np.zeros((Nm, 3))
        c = np.zeros((Nm, 3))
        for d in range(3):
            core = np.pi * j[:, d] * (X[d, i] - a[d]) / (b[d] - a[d])
            mult = 1. / np.sqrt(0.5 * (b[d] - a[d]))
            s[:, d] = np.sin(core) * mult
            c[:, d] = np.cos(core) * mult

        J[0, 0, :, i] = -f[:, 0] ** 2 * s[:, 0] * s[:, 1] * s[:, 2]
        J[0, 1, :, i] = f[:, 0] * f[:, 1] * c[:, 0] * c[:, 1] * s[:, 2]
        J[0, 2, :, i] = f[:, 0] * f[:, 2] * c[:, 0] * s[:, 1] * c[:, 2]
        J[1, 0, :, i] = f[:, 1] * f[:, 0] * c[:, 0] * c[:, 1] * s[:, 2]
        J[1, 1, :, i] = -f[:, 1] ** 2 * s[:, 0] * s[:, 1] * s[:, 2]
        J[1, 2, :, i] = f[:, 1] * f[:, 2] * s[:, 0] * c[:, 1] * c[:, 2]
        J[2, 0, :, i] = f[:, 2] * f[:, 0] * c[:, 0] * s[:, 1] * c[:, 2]
        J[2, 1, :, i] = f[:, 2] * f[:, 1] * s[:, 0] * c[:, 1] * c[:, 2]
        J[2, 2, :, i] = -f[:, 2] ** 2 * s[:, 0] * s[:, 1] * s[:, 2]

    return J

def spectralSE(omega, modelParameters):
    ''' Copyright (C) 2022 by Frida Viset
        Adapted into python by Thomas Edridge
        This function is implemented based on
        Equation 21 in "Modeling and Interpolation
        of the Ambient Magnetic Field by Gaussian
        Processes" by Manon Kok and Arno Solin,
        published in IEEE TRANSACTIONS ON ROBOTICS,
        VOL. 34, NO. 4, AUGUST 2018 '''
    theta = modelParameters['theta']
    return theta[1] * (2 * np.pi * theta[0])**(3/2) * np.exp(-omega**2 * theta[0] / 2);

def predictReducedRankCurlFree(Xdata, Xtest, ydata, Nm, indices, Lambda, modelParameters):
    theta = modelParameters['theta']
    phiMeas = np.squeeze(nablaPhi3D(Xdata, Nm, indices, modelParameters))
    phiPredict = np.squeeze(nablaPhi3D(Xtest, Nm, indices, modelParameters))
    fpredict = phiPredict @ linAlg.sinv(phiMeas.T @ phiMeas + theta[2] * Lambda)
    iota = phiMeas.T @ ydata.T.reshape(-1, 1)
    fpredict = fpredict @ iota
    covPredict =  theta[2] * phiPredict @ linAlg.sinv(phiMeas.T @ phiMeas + theta[2] * Lambda) @ phiPredict.T
    return fpredict.reshape(-1, 3).T, covPredict
