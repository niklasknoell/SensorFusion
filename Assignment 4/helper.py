import copy
import numpy as np
import matplotlib.pyplot as plt
import os as os
import linAlg as linAlg


def initializeDKFAssignment():
    # Load the dataset
    myPath = os.getcwd()
    myFile = "tracking_data.npz"
    data = np.load(os.path.join(myPath, myFile))

    # Create dictionary to store parameters
    model_parameters = {
        "mp": data["mp"],
    }

    dt = data["dt"]
    T = data["T"]
    anchor_positions = data["anchor_positions"]
    edge_list = data["edge_list"]
    measurements = data["measurement_list"]
    true_states = data["true_states"]

    return (
        measurements,
        true_states,
        anchor_positions,
        edge_list,
        dt,
        T,
        model_parameters,
    )


def plotMeasurementsAndGroundTruth(
    true_states, anchor_positions, edge_list, model_parameters
):
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    num_anchors = anchor_positions.shape[1]  # Number of sensors/anchors

    mp = model_parameters["mp"]

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.plot(
        true_states[0, :],
        true_states[1, :],
        label=f"True Positions",
        ls="-.",
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
        color="black",
    )
    for e in edge_list:
        edge = np.vstack((anchor_positions[:, e[0]], anchor_positions[:, e[1]])).T
        plt.plot(edge[0, :], edge[1, :], ls="--", color=[0.1, 0.1, 0.1, 0.5])
    for i in range(num_anchors - 1):
        plt.plot(
            mp[i][0, :],
            mp[i][1, :],
            ls="",
            marker="x",
            markerfacecolor="red",
            markeredgecolor="red",
            color="red",
            markersize=6,
        )
    plt.plot(
        mp[num_anchors - 1][0, :],
        mp[num_anchors - 1][1, :],
        ls="",
        marker="x",
        markerfacecolor="red",
        markeredgecolor="red",
        color="red",
        markersize=6,
        label="Measurements",
    )
    plt.plot(
        anchor_positions[0, :],
        anchor_positions[1, :],
        label=f"Anchor Positions",
        ls="",
        marker="o",
        color=color_cycle[0],
        markeredgecolor="black",
    )
    plt.xlabel("Horizontal Distance [m]")
    plt.ylabel("Vertical Distance [m]")
    plt.legend()
    plt.grid()
    plt.gca().set_aspect("equal")
    plt.show()


def plotCentralizedKFEstimate(true_states, anchor_positions, edge_list, estimates):
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    estimates = copy.deepcopy(estimates)
    estimates = np.hstack(estimates)

    plt.figure(figsize=(8, 8))
    plt.plot(
        true_states[0, :],
        true_states[1, :],
        label=f"True Positions",
        ls="-.",
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
        color="black",
    )
    for e in edge_list:
        edge = np.vstack((anchor_positions[:, e[0]], anchor_positions[:, e[1]])).T
        plt.plot(edge[0, :], edge[1, :], ls="--", color=[0.1, 0.1, 0.1, 0.8])
    plt.plot(
        estimates[0, :],
        estimates[1, :],
        ls="-.",
        marker="+",
        markeredgecolor=color_cycle[1],
        color=color_cycle[1],
        markersize=8,
        label="Centralized Estimates",
    )
    plt.plot(
        anchor_positions[0, :],
        anchor_positions[1, :],
        label=f"Anchor Positions",
        ls="",
        marker="o",
        color=color_cycle[0],
        markeredgecolor="black",
    )
    plt.xlabel("Horizontal Distance [m]")
    plt.ylabel("Vertical Distance [m]")
    plt.legend()
    plt.grid()
    plt.gca().set_aspect("equal")
    plt.show()


def plotDecentralizedKFEstimate(true_states, anchor_positions, edge_list, estimates):
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    num_anchors = anchor_positions.shape[1]
    estimates = copy.deepcopy(estimates)

    for i in range(num_anchors):
        estimates[i] = np.hstack(estimates[i])

    plt.figure(figsize=(8, 8))
    plt.plot(
        true_states[0, :],
        true_states[1, :],
        label=f"True Positions",
        ls="-.",
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
        color="black",
    )
    for e in edge_list:
        edge = np.vstack((anchor_positions[:, e[0]], anchor_positions[:, e[1]])).T
        plt.plot(edge[0, :], edge[1, :], ls="--", color=[0.1, 0.1, 0.1, 0.8])
    for i in range(num_anchors):
        plt.plot(
            estimates[i][0, :],
            estimates[i][1, :],
            ls="-.",
            marker="+",
            markeredgecolor=color_cycle[i],
            color=color_cycle[i],
            markersize=8,
            label=f"Local Estimate {i}",
        )
    plt.plot(
        anchor_positions[0, :],
        anchor_positions[1, :],
        label=f"Anchor Positions",
        ls="",
        marker="o",
        color=color_cycle[0],
        markeredgecolor="black",
    )
    plt.xlabel("Horizontal Distance [m]")
    plt.ylabel("Vertical Distance [m]")
    plt.legend()
    plt.grid()
    plt.gca().set_aspect("equal")
    plt.show()
