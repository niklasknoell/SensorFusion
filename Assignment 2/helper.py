import numpy as np
import matplotlib.pyplot as plt
import os as os
import linAlg as linAlg
import GP as GP
from scipy.interpolate import griddata
from matplotlib.patches import Ellipse


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


def initializeKalmanFilterAssignment(groupNumber):
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
   
def makeDeadReckoningPlots(deadReckoning, magneticFieldNorm, modelParameters):
    ''' Plot settings '''
    startPlot = -.4
    endPlot = .4

    predictionDomain = np.array([[startPlot, endPlot], 
                    [startPlot, endPlot], 
                    [startPlot, endPlot]])  

    predictionLocations = linAlg.gridpointsHyperCube(modelParameters['Npred'], 2, 3, predictionDomain)
    predictionMagneticField = GP.makeMagneticFieldPrediction(predictionLocations, modelParameters)
    posPlot1, posPlot2, fPlotNorm = makeIntoCountour(predictionLocations, predictionMagneticField)

    fig, axs = plt.subplots(1, 2, figsize=(12, 7))

    Vmin = np.min([np.min(magneticFieldNorm), np.min(fPlotNorm)])
    Vmax = np.max([np.max(magneticFieldNorm), np.max(fPlotNorm)])
    sc = axs[1].contourf(posPlot1, posPlot2, fPlotNorm, vmin=Vmin, vmax=Vmax, levels=25, cmap='viridis')

    axs[1].scatter(deadReckoning[0,:], deadReckoning[1,:], c = magneticFieldNorm, vmin=Vmin, vmax=Vmax, label='Dead Reckoning')
    axs[1].set_xlabel('x-position [m]')
    axs[1].set_ylabel('y-position [m]')
    axs[1].axis([startPlot, endPlot, startPlot, endPlot])
    axs[1].set_title('Dead Reckoning and measurement norm')
    
    
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
    plt.show()
    return

def plot_covariance_ellipse(ax, mean, cov, nstd=2, **kwargs):
    """
    Plots an ellipse representing the covariance matrix.
    
    Parameters:
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    mean : array-like, shape (2,)
        The mean of the distribution.
    cov : array-like, shape (2, 2)
        The covariance matrix.
    nstd : int
        The number of standard deviations to determine the ellipse's radii.
    kwargs : dict
        Additional arguments to pass to Ellipse.
    """
    # eigvals, eigvecs = np.linalg.eigh(cov)
    # order = eigvals.argsort()[::-1]
    # eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    # angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    # width, height = 2 * nstd * np.sqrt(eigvals)
    # ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    # ax.add_patch(ellipse)
    scale_num = 1.0
    eigvals, eigvecs = np.linalg.eigh(cov*scale_num)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    # Ensure that the inputs to np.arctan2 are arrays
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    # Calculate the width and height of the ellipse
    width, height = 2 * nstd * np.sqrt(eigvals)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return


def makeExtendedKalmanFilterPlots(deadReckoning, meanEKFPosition, covEKFPosition, modelParameters):
        
    ''' Plot settings '''
    startPlot = -.4
    endPlot = .4

    predictionDomain = np.array([[startPlot, endPlot], 
                    [startPlot, endPlot], 
                    [startPlot, endPlot]])  

    predictionLocations = linAlg.gridpointsHyperCube(modelParameters['Npred'], 2, 3, predictionDomain)
    predictionMagneticField = GP.makeMagneticFieldPrediction(predictionLocations, modelParameters)

    posPlot1, posPlot2, fPlotNorm = makeIntoCountour(predictionLocations, predictionMagneticField)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    sc = ax1.contourf(posPlot1, posPlot2, fPlotNorm, levels=25, cmap='viridis')
    ax1.plot(modelParameters['posGroundTruth'][0, :], modelParameters['posGroundTruth'][1, :], c='black', label='True Trajectory')
    ax1.plot(deadReckoning[0, :], deadReckoning[1, :], color='red', label='Dead reckoning')
    ax1.plot(meanEKFPosition[0, :], meanEKFPosition[1, :], c='white', label='EKF Trajectory')
    # add the covariance ellipse
    for i in range(meanEKFPosition.shape[1]):
        CovEKF = np.zeros((2,2))
        CovEKF[0,0] = covEKFPosition[0,0,i]
        CovEKF[1,1] = covEKFPosition[1,1,i]
        MeanEKF = meanEKFPosition[0:2, i]
        # plot_covariance_ellipse(ax1, MeanEKF, CovEKF, edgecolor='blue', alpha=0.5)   
        plot_covariance_ellipse(ax1, MeanEKF, CovEKF, edgecolor='blue', facecolor='blue', alpha=0.2, linewidth=2)

    ax1.set_xlim([startPlot, endPlot])
    ax1.set_ylim([startPlot, endPlot])
    ax1.set_xlabel('x-position [m]')
    ax1.set_ylabel('y-position [m]')
    ax1.legend(loc='upper left')
    ax1.set_title('EKF Trajectory with Covariance Ellipses')

    # Plot value-time curve for covariance
    time_steps = np.arange(covEKFPosition.shape[2])
    cov_x = covEKFPosition[0, 0, :]
    cov_y = covEKFPosition[1, 1, :]
    cov_z = covEKFPosition[2, 2, :]

    ax2.plot(time_steps, cov_x, label='Covariance X', color='blue')
    ax2.plot(time_steps, cov_y, label='Covariance Y', color='green')
    ax2.plot(time_steps, cov_z, label='Covariance Z', color='red')
    ax2.set_xlabel('Time step [t]')
    ax2.set_ylabel(r'Covariance [(\mu T)^2]')
    ax2.legend()
    ax2.set_title('Covariance over Time')

    
    cbar = fig.colorbar(sc, ax=[ax1, ax2], orientation='horizontal', fraction=0.1, pad=0.1)
    cbar.set_label(r'Norm magnetic field [$\mu$ T]')

    plt.show()
    return


def makeUnscentedKalmanFilterPlots(deadReckoning, meanUKFPosition, covUKFPosition, modelParameters):
    ''' Plot settings '''
    startPlot = -.4
    endPlot = .4

    predictionDomain = np.array([[startPlot, endPlot], 
                    [startPlot, endPlot], 
                    [startPlot, endPlot]])  

    predictionLocations = linAlg.gridpointsHyperCube(modelParameters['Npred'], 2, 3, predictionDomain)
    predictionMagneticField = GP.makeMagneticFieldPrediction(predictionLocations, modelParameters)

    posPlot1, posPlot2, fPlotNorm = makeIntoCountour(predictionLocations, predictionMagneticField)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

    sc = ax1.contourf(posPlot1, posPlot2, fPlotNorm, levels=25, cmap='viridis')
    ax1.plot(modelParameters['posGroundTruth'][0, :], modelParameters['posGroundTruth'][1, :], c='black', label='True Trajectory')
    ax1.plot(deadReckoning[0, :], deadReckoning[1, :], color='red', label='Dead reckoning')
    ax1.plot(meanUKFPosition[0, :], meanUKFPosition[1, :], c='white', label='UKF Trajectory')
    
    # Add the covariance ellipses
    for i in range(meanUKFPosition.shape[1]):
        CovUKF = np.zeros((2,2))
        CovUKF[0,0] = covUKFPosition[0,0,i]
        CovUKF[1,1] = covUKFPosition[1,1,i]
        MeanUKF = meanUKFPosition[0:2, i]
        plot_covariance_ellipse(ax1, MeanUKF,CovUKF, edgecolor='blue', alpha=0.5)

    ax1.set_xlim([startPlot, endPlot])
    ax1.set_ylim([startPlot, endPlot])
    ax1.set_xlabel('x-position [m]')
    ax1.set_ylabel('y-position [m]')
    ax1.legend(loc='upper left')
    ax1.set_title('UKF Trajectory with Covariance Ellipses')

    cbar = fig.colorbar(sc, ax=ax1, orientation='horizontal', fraction=0.1, pad=0.1)
    cbar.set_label(r'Norm magnetic field [$\mu$ T]')

    plt.show()
    return


def makeKalmanFilterPlots(deadReckoning, meanEKFPosition, meanUKFPosition, modelParameters):
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

    # Plot EKF Trajectory
    sc = ax1.contourf(posPlot1, posPlot2, fPlotNorm, levels=25, cmap='viridis')
    ax1.plot(modelParameters['posGroundTruth'][0, :], modelParameters['posGroundTruth'][1, :], c='black', label='True Trajectory')
    ax1.plot(deadReckoning[0, :], deadReckoning[1, :], color='red', label='Dead reckoning')
    ax1.plot(meanEKFPosition[0, :], meanEKFPosition[1, :], c='white', label='EKF Trajectory')

    ax1.set_xlim([startPlot, endPlot])
    ax1.set_ylim([startPlot, endPlot])
    ax1.set_xlabel('x-position [m]')
    ax1.set_ylabel('y-position [m]')
    ax1.legend(loc='upper left')
    ax1.set_title('EKF Trajectory')

    # Plot UKF Trajectory
    sc = ax2.contourf(posPlot1, posPlot2, fPlotNorm, levels=25, cmap='viridis')
    ax2.plot(modelParameters['posGroundTruth'][0, :], modelParameters['posGroundTruth'][1, :], c='black', label='True Trajectory')
    ax2.plot(deadReckoning[0, :], deadReckoning[1, :], color='red', label='Dead reckoning')
    ax2.plot(meanUKFPosition[0, :], meanUKFPosition[1, :], c='white', label='UKF Trajectory')

    ax2.set_xlim([startPlot, endPlot])
    ax2.set_ylim([startPlot, endPlot])
    ax2.set_xlabel('x-position [m]')
    ax2.set_ylabel('y-position [m]')
    ax2.legend(loc='upper left')
    ax2.set_title('UKF Trajectory')

    cbar = fig.colorbar(sc, ax=[ax1, ax2], orientation='horizontal', fraction=0.1, pad=0.1)
    cbar.set_label(r'Norm magnetic field [$\mu$ T]')

    plt.show()
    return