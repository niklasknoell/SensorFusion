import numpy as np
import matplotlib.pyplot as plt
import os as os
import linAlg as linAlg
import GP as GP
from scipy.interpolate import griddata

def initializeGaussianProcessAssignment(groupNumber):
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
    magnetometerPositions = data['posGroundTruth']
    return magnetometerMeasurements, magnetometerPositions, modelParameters

def makeIntoCountour(posPred, fPredNorm, Nplot = 50):
    startPred = np.min(np.min(posPred))
    endPred = np.max(np.max(posPred))
    P0 = np.linspace(startPred, endPred, Nplot)
    P1 = np.linspace(startPred, endPred, Nplot)
    P0plot, P1plot = np.meshgrid(P0, P1)
    # fPredNorm = np.linalg.norm(fPred, axis=0).reshape(1, -1)
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

def makeGaussianProcessMagneticFieldMapPlots(magnetometerPositions, posPred, magneticFieldPredictions, magneticFieldPredictionsCovariance, magnetomerNorm, modelParameters):
    if magneticFieldPredictions.shape[1] == 1:
        magneticFieldPredictions = magneticFieldPredictions.T

    ''' Plot settings '''
    startPlot = -.4
    endPlot = .4

    predictionDomain = np.array([[startPlot, endPlot], 
                    [startPlot, endPlot], 
                    [startPlot, endPlot]])  

    predictionLocations = linAlg.gridpointsHyperCube(modelParameters['Npred'], 2, 3, predictionDomain)
    predictionMagneticField = GP.makeMagneticFieldPrediction(predictionLocations, modelParameters)

    predictionMagneticFieldNorm = linAlg.vectorToScalarNorm(predictionMagneticField)
    posPlotX1, posPlotY1, fPlotNorm1 = makeIntoCountour(predictionLocations, predictionMagneticFieldNorm)
    posPlotX2, posPlotY2, fPlotNormCov2 = makeIntoCountour(posPred, np.diag(magneticFieldPredictionsCovariance).reshape(1, -1))
    posPlotX2, posPlotY2, fPlotNorm2 = makeIntoCountour(posPred, magneticFieldPredictions)
    fPlotNormCov2 = linAlg.normaliseArray(fPlotNormCov2)

    Vmin = np.min([np.min(fPlotNorm1), np.min(fPlotNorm2), np.min(magnetomerNorm)])
    Vmax = np.max([np.max(fPlotNorm1), np.max(fPlotNorm2), np.max(magnetomerNorm)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    sc = ax1.contourf(posPlotX1, posPlotY1, fPlotNorm1, levels=50, cmap='viridis', vmin=Vmin, vmax=Vmax)
#     ax1.plot(magnetometerPositions[0, :], magnetometerPositions[1, :], c = 'black', label = 'magnetometer positions')
    ax1.scatter(magnetometerPositions[0, :], magnetometerPositions[1, :], c = np.squeeze(magnetomerNorm), vmin=Vmin, vmax=Vmax, label = 'magnetometer measurements')
    ax1.set_xlim([startPlot, endPlot])
    ax1.set_ylim([startPlot, endPlot])
    ax1.set_xlabel('x-position [m]')
    ax1.set_ylabel('y-position [m]')
    ax1.legend(loc='lower center', ncol=3, fontsize='small')
    ax1.set_title('Reference magnetic field map')
    

    sc = ax2.contourf(posPlotX2, posPlotY2, fPlotNorm2, levels=50, cmap='viridis', vmin=Vmin, vmax=Vmax)
#     ax2.plot(magnetometerPositions[0, :], magnetometerPositions[1, :], c = 'black', label = 'magnetometer positions')
    ax2.scatter(magnetometerPositions[0, :], magnetometerPositions[1, :], c = np.squeeze(magnetomerNorm), vmin=Vmin, vmax=Vmax, label = 'magnetometer measurements')
    ax2.set_xlim([startPlot, endPlot])
    ax2.set_ylim([startPlot, endPlot])
    ax2.set_xlabel('x-position [m]')
    ax2.set_ylabel('y-position [m]')
    ax2.legend(loc='lower center', ncol=3, fontsize='small')
    ax2.set_title('Estimated magnetic field map')

    cbar = fig.colorbar(sc, ax=[ax1, ax2], orientation='horizontal', fraction=0.1, pad=0.1)
    cbar.set_label(r'Norm magnetic field [$\mu$ T]')

    plt.show()
    return





def generateData(posData1D, functionName, addNoise = True):
    posData1D = posData1D.reshape(-1, 1)
    if functionName == 'sin':
        fData1D = np.sin(posData1D)
    elif functionName == 'linsin':
        fData1D = np.sin(posData1D)
        fData1D += 0.25 * posData1D
        
    if addNoise == True:
        return fData1D.reshape(-1, 1) + np.random.normal(0, np.sqrt(0.01), len(posData1D)).reshape(-1, 1)
    else:
        return fData1D.reshape(-1, 1)
    


def makeGaussianProcessPredictionPlots(posData1D, yData1D, posPred1D, f, cov, functionName = 'none', noiseVarianceOpt = 0):
    std_dev = np.sqrt(np.diag(cov)+noiseVarianceOpt).reshape(-1, 1) 
    fig = plt.figure(figsize = (9,3), dpi=100)
    plt.fill_between(posPred1D.flatten(), (f - 2 * std_dev).flatten(), (f + 2 * std_dev).flatten(), color='royalblue', alpha=0.125, label = '2 STD')
    
    numSamples = 50
    samples = np.random.multivariate_normal(f.flatten(), cov, numSamples)
    for i in range(numSamples):
        if i == 0:
            plt.plot(posPred1D.flatten(), samples[i], color='royalblue', alpha =.15, label = 'GP samples')
        else:
            plt.plot(posPred1D.flatten(), samples[i], color='royalblue', alpha =.15)

    plt.plot(posPred1D.T, f, color='royalblue', label = 'Mean prediction')
    if functionName == 'none':
        latentF = f
        pass
    else:
        latentX = np.linspace(-20, 20, 250).reshape(1, -1)
        latentF = generateData(latentX, functionName, addNoise = False)
        plt.plot(latentX.flatten(), latentF.flatten(), color='black', linestyle=':', label = 'latent function')

    plt.scatter(posData1D, yData1D, color='red', label = 'Measurement')


    
    plt.axis([np.min(posPred1D), np.max(posPred1D), np.min([np.min(f),np.min(latentF)]) - 2, np.max([np.max(f),np.max(latentF)]) + 2])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5)
    plt.xlabel('input x [-]')
    plt.ylabel('output f [-]')
    plt.show()






def makeGaussianProcessSamplingPlots(posPred1D, fSamples, posData1D = 0, yData1D = 0):
    fig = plt.figure(figsize = (9,3), dpi=100)
    nColumns = 1
    numSamples = fSamples.shape[0]
    for i in range(numSamples):
        if i == 0:
            plt.plot(posPred1D.flatten(), fSamples[i, :], color='royalblue', alpha =.3, label = 'GP samples')
        else:
            plt.plot(posPred1D.flatten(), fSamples[i, :], color='royalblue', alpha =.3)
    # if f != 0:
    #     plt.plot(posPred1D.T, f, color='royalblue', label = 'Mean prediction')
    #     nColumns += 1
    if isinstance(posData1D, np.ndarray) and isinstance(yData1D, np.ndarray):
        plt.scatter(posData1D, yData1D, color='red', label = 'Measurement')
        nColumns += 1
    plt.axis([np.min(posPred1D), np.max(posPred1D), np.min(fSamples) - 2, np.max(fSamples) + 2])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=nColumns)
    plt.xlabel('input x [-]')
    plt.ylabel('output f [-]')
    plt.show()



# def makeGaussianProcessnPlots(posData1D, posPred1D, yData1D, f, functionName = 'none', noiseVarianceOpt = 0):
#     fig = plt.figure(figsize = (9,3), dpi=100)
    
#     numSamples = 25
#     samples = np.random.multivariate_normal(f.flatten(), cov, numSamples)
#     for i in range(numSamples):
#         if i == 0:
#             plt.plot(posPred1D.flatten(), samples[i], color='royalblue', alpha =.15, label = 'GP samples')
#         else:
#             plt.plot(posPred1D.flatten(), samples[i], color='royalblue', alpha =.15)

#     plt.plot(posPred1D.T, f, color='royalblue', label = 'Mean prediction')
#     if functionName == 'none':
#         pass
#     else:
#         latentX = np.linspace(-20, 20, 250).reshape(1, -1)
#         latentF = generateData(latentX, functionName, addNoise = False)
#         plt.plot(latentX.flatten(), latentF.flatten(), color='black', linestyle=':', label = 'latent function')

#     plt.scatter(posData1D, yData1D, color='red', label = 'Measurement')

    
#     plt.axis([np.min(posPred1D), np.max(posPred1D), np.min(f) - 2, np.max(f) + 2])
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=5)
#     plt.xlabel('input x [-]')
#     plt.ylabel('output f [-]')
#     plt.show()