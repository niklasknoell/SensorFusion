import numpy as np
import matplotlib.pyplot as plt
import os as os
import linAlg as linAlg
import GP as GP
from scipy.interpolate import griddata

def initializeParticleFilterAssignment(groupNumber):
    ''' load the dataset '''
    myPath = os.getcwd()
    myData = 'processedData'
    myFile = 'magMeas' + str(groupNumber) + '.npz'
    data = np.load(os.path.join(myPath, myData, myFile))

    ''' Create dictionary to store parameters '''
    modelParameters = {
        "theta": data['theta'],
        "Din": data['Din'],
        "Npred": data['Npred'],
        "NtimeSteps": data['magMeas'].shape[1],
        "domain": data['domain'],
        "margin": data['margin'],
        "GPweights": data['GPweights'],
        "GPdataInverse": data['GPdataInverse'],
        "numberOfBasisFunctions": data['numberOfBasisFunctions'],
        "posGroundTruth": data['posGroundTruth'],
    }

    ''' Number of basis functions used in Reduced-Rank approximation'''
    GP.setNumberOfBasisFunctions(data['numberOfBasisFunctions'], modelParameters)

    ''' Loading the measurements data, 
        You will use these in the particle filter '''
    magnetometerMeasurements = data['magMeas']
    magnetometerInitialPosition = data['magnetometerInitialPosition']
    deltaMagnetometerPositions = data['deltaMagLoc']
    return magnetometerMeasurements, magnetometerInitialPosition, deltaMagnetometerPositions, modelParameters

def makeIntoCountour(posPred, fPred, Nplot = 50):
    startPred = np.min(np.min(posPred))
    endPred = np.max(np.max(posPred))
    P0 = np.linspace(startPred, endPred, Nplot)
    P1 = np.linspace(startPred, endPred, Nplot)
    P0plot, P1plot = np.meshgrid(P0, P1)
    fPredNorm = linAlg.vectorToScalarNorm(fPred)
    fPlotNorm = griddata((posPred[0, :], posPred[1, :]), fPredNorm[0, :], (P0plot, P1plot), method="cubic")
    return P0plot, P1plot, fPlotNorm
   
def makeInitialPositionPlots(deadReckoning, magneticFieldNorm, modelParameters):
    ''' Plot settings '''
    startPlot = -.4
    endPlot = .4

    predictionDomain = np.array([[startPlot, endPlot], 
                    [startPlot, endPlot], 
                    [startPlot, endPlot]])  

    predictionLocations = linAlg.gridpointsHyperCube(modelParameters['Npred'], 2, 3, predictionDomain)
    predictionMagneticField = GP.makeMagneticFieldPrediction(predictionLocations, modelParameters)
    posPlot1, posPlot2, fPlotNorm = makeIntoCountour(predictionLocations, predictionMagneticField)

    fig, axs = plt.subplots(1, 3, figsize=(18, 7))

    Vmin = np.min([np.min(magneticFieldNorm), np.min(fPlotNorm)])
    Vmax = np.max([np.max(magneticFieldNorm), np.max(fPlotNorm)])


    sc = axs[0].contourf(posPlot1, posPlot2, fPlotNorm, vmin=Vmin, vmax=Vmax, levels=25, cmap='viridis')
    cbar = fig.colorbar(sc, ax=axs, orientation='horizontal', fraction=0.1, pad=0.1)
    cbar.set_label(r'Norm magnetic field [$\mu$ T]')

    axs[0].plot(deadReckoning[0,:], deadReckoning[1,:], 'r', label='Dead Reckoning')
    axs[0].plot(modelParameters['posGroundTruth'][0, :], modelParameters['posGroundTruth'][1, :], c='black', label='True Trajectory')
    axs[0].set_xlabel('x-position [m]')
    axs[0].set_ylabel('y-position [m]')
    axs[0].axis([startPlot, endPlot, startPlot, endPlot])
    axs[0].set_title('Ground truth position vs Dead Reckoning')
    axs[0].legend()

    sc = axs[1].contourf(posPlot1, posPlot2, fPlotNorm, vmin=Vmin, vmax=Vmax, levels=25, cmap='viridis')

    axs[1].scatter(modelParameters['posGroundTruth'][0,:], modelParameters['posGroundTruth'][1,:], c = magneticFieldNorm, vmin=Vmin, vmax=Vmax)
    axs[1].set_xlabel('x-position [m]')
    axs[1].set_ylabel('y-position [m]')
    axs[1].axis([startPlot, endPlot, startPlot, endPlot])
    axs[1].set_title('Ground truth positions and measurement norm')
    
    
    sc = axs[2].contourf(posPlot1, posPlot2, fPlotNorm, vmin=Vmin, vmax=Vmax, levels=25, cmap='viridis')

    axs[2].scatter(deadReckoning[0,:], deadReckoning[1,:], c = magneticFieldNorm, vmin=Vmin, vmax=Vmax)
    axs[2].set_xlabel('x-position [m]')
    axs[2].set_ylabel('y-position [m]')
    axs[2].axis([startPlot, endPlot, startPlot, endPlot])
    axs[2].set_title('Dead Reckoning positions and measurement norm')
    

    plt.show()

def makeParticleFilterPlots(deadReckoning, meanParticlePosition, bestParticlePosition, modelParameters):
        
    ''' Plot settings '''
    startPlot = -.4
    endPlot = .4

    predictionDomain = np.array([[startPlot, endPlot], 
                    [startPlot, endPlot], 
                    [startPlot, endPlot]])  

    predictionLocations = linAlg.gridpointsHyperCube(modelParameters['Npred'], 2, 3, predictionDomain)
    predictionMagneticField = GP.makeMagneticFieldPrediction(predictionLocations, modelParameters)

    posPlot1, posPlot2, fPlotNorm = makeIntoCountour(predictionLocations, predictionMagneticField)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    sc = ax1.contourf(posPlot1, posPlot2, fPlotNorm, levels=25, cmap='viridis')
    ax1.plot(modelParameters['posGroundTruth'][0, :], modelParameters['posGroundTruth'][1, :], c='black', label='True Trajectory')
    ax1.plot(deadReckoning[0, :], deadReckoning[1, :], color='red', label='Dead reckoning')
    ax1.plot(meanParticlePosition[0, :], meanParticlePosition[1, :], c='white', label='Weighted Average Particle Trajectory')

    ax1.set_xlim([startPlot, endPlot])
    ax1.set_ylim([startPlot, endPlot])
    ax1.set_xlabel('x-position [m]')
    ax1.set_ylabel('y-position [m]')
    ax1.legend(loc='lower center', ncol=3, fontsize='small')
    ax1.set_title('Weighted average particle trajectory')


    sc = ax2.contourf(posPlot1, posPlot2, fPlotNorm, levels=25, cmap='viridis')

    ax2.plot(modelParameters['posGroundTruth'][0, :], modelParameters['posGroundTruth'][1, :], c='black', label='True Trajectory')
    ax2.plot(deadReckoning[0, :], deadReckoning[1, :], color='red', label='Dead reckoning')
    ax2.plot(bestParticlePosition[0, :], bestParticlePosition[1, :], c='white', label='Best Particle Trajectory')

    ax2.set_xlim([startPlot, endPlot])
    ax2.set_ylim([startPlot, endPlot])
    ax2.set_xlabel('x-position [m]')
    ax2.set_ylabel('y-position [m]')
    ax2.legend(loc='lower center', ncol=3, fontsize='small')
    ax2.set_title('Best particle trajectory')

    cbar = fig.colorbar(sc, ax=[ax1, ax2], orientation='horizontal', fraction=0.1, pad=0.1)
    cbar.set_label(r'Norm magnetic field [$\mu$ T]')

    plt.show()
    return

def systematicResample(particleWeights):
    ''' Based on the Matlab code written by:
        Thomas Schön (schon@isy.liu.se)
        Division of Automatic Control
        Linköping University
        www.control.isy.liu.se/~schon
        Last revised on September 30, 2010 '''
    wc = np.cumsum(particleWeights)
    M = len(particleWeights)
    u = (np.arange(0, M) + np.random.rand(1)) / M
    index = np.zeros(M, dtype=int)
    k = 0
    for j in range(M):
        while wc[k] < u[j]:
            k += 1
        index[j] = k
    return index