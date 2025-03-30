"""
This file contains necessary utility functions for the assignment. 
Please do not modify the code in this file.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def param_array_to_dict(param_array):
    """
    Converts a 1D numpy array to a formated dictionary of hyperparameters
    """
    return {
        'signal_var': param_array[0],
        'length_scale': np.array([param_array[1], param_array[2]]),
        'noise_std': param_array[3]
    }

def param_dict_to_array(param_dict):
    """
    Converts a dictionary of hyperparameters to a 1D numpy array
    """

    return np.array([param_dict['signal_var'], param_dict['length_scale'][0], param_dict['length_scale'][1], param_dict['noise_std']])


def data_generator(sample_locations, gp_field, field_range, field_resolution):
    """
    Generates clean samples from the GP field at the given sample locations
    - sample_locations: 2D numpy array of shape (INPUT_DIM, NUM_SAMPLES)
    """

    x1 = np.linspace(field_range[0,0], field_range[0,1], field_resolution[0])
    x2 = np.linspace(field_range[1,0], field_range[1,1], field_resolution[1])

    interp_func = RegularGridInterpolator((x1, x2), gp_field)
    clean_samples = interp_func(sample_locations.T)

    mesh_x1, mesh_x2 = np.meshgrid(x1, x2, indexing='ij')

    return clean_samples, mesh_x1, mesh_x2

def visualize_samples_on_field(ax, mesh_x1, mesh_x2, gp_field, sample_locations, samples):
    """
    Visualizes the GP field and the samples
    """
    # plt.figure(figsize=(8, 6))

    # Plot the original grid data
    contour = ax.contourf(mesh_x1, mesh_x2, gp_field, levels=50, cmap='plasma', alpha=0.7)
    # ax.colorbar(label='Field Value')

    # Overlay agents and sample locations and their interpolated values
    ax.scatter(sample_locations[0,:], sample_locations[1, :], c=samples, s=50, cmap='plasma', edgecolor='k', label='Sample Points')

    # Labels and title
    ax.set_title("GP Field")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    # plt.legend(loc='upper right')
    # plt.show()
    return contour

def visualize_sample_allocation(ax, agent_locations, sample_locations_list: list, adjacency_matrix):
    """
    Visualizes the sample allocation among agents
    """
    num_agents = agent_locations.shape[1]
    colors = plt.cm.get_cmap("tab10", num_agents)  # Use a colormap with N unique colors

    # Plot the agents
    # plt.figure(figsize=(6, 6))
    for i in range(num_agents):
        ax.scatter(agent_locations[0, i], agent_locations[1, i], marker = 'x', s = 70, color=colors(i))
        ax.scatter(sample_locations_list[i][0,:], sample_locations_list[i][1,:], marker = 'o', color=colors(i))

    # Plot edges
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if adjacency_matrix[i, j] == 1:
                ax.plot([agent_locations[0, i], agent_locations[0, j]],
                         [agent_locations[1, i], agent_locations[1, j]],
                         'gray', linewidth=1, zorder=1)
    
    ax.set_aspect("equal")
    # plt.show()

def visualize_graphs(agent_locations , adjacency_matrix):
    """
    Visualizes the underlying graph
    """
    num_agents = agent_locations.shape[1]
    colors = plt.cm.get_cmap("tab10", num_agents)  # Use a colormap with N unique colors

    # Plot the agents
    plt.figure(figsize=(3, 3))
    for i in range(num_agents):
        plt.scatter(agent_locations[0, i], agent_locations[1, i], marker = 'x', s = 70, color=colors(i))
        # plt.scatter(sample_locations_list[i][0,:], sample_locations_list[i][1,:], marker = 'o', color=colors(i))

    # Plot edges
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if adjacency_matrix[i, j] == 1:
                plt.plot([agent_locations[0, i], agent_locations[0, j]],
                         [agent_locations[1, i], agent_locations[1, j]],
                         'gray', linewidth=1, zorder=1)
    
    plt.axis("equal")
    plt.show()
