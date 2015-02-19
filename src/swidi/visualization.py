# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np

from discretizations import DiscretizationInterface


def visualize_trajectory_1d(trajectory, states=None, discretization=None):
    assert isinstance(trajectory, np.ndarray) and len(trajectory.shape) == 2 and trajectory.shape[1] == 1
    assert states is None or (isinstance(states, np.ndarray) and states.shape[0] == trajectory.shape[0])
    assert discretization is None or isinstance(discretization, DiscretizationInterface)

    n = trajectory.shape[0]
    if discretization is None:
        x = np.arange(n)
    else:
        x = discretization.time_steps

    y = trajectory.ravel()

    if states is not None:
        plt.subplot(2, 1, 2)
        plt.plot(x, states)
        plt.title("Markov parameter r(t)")
        plt.ylim(min(states)-0.1, max(states)+0.1)
        plt.subplot(2, 1, 1)

    plt.plot(x, y)
    plt.title("State X(t)")

    plt.show()
