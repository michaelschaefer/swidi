# -*- coding: utf-8 -*-


import numpy as np
from scipy.linalg import expm
import warnings

from la import OperatorInterface


class DiscreteTimeMarkovChain(object):

    def __init__(self, number_of_states, initial_state, generator):
        assert isinstance(number_of_states, int) and number_of_states > 0
        assert isinstance(initial_state, int) and 0 <= initial_state <= number_of_states
        assert isinstance(generator, OperatorInterface)

        self.states = [i for i in range(number_of_states)]
        self.current_state = initial_state
        self.dt = None

        assert generator.range.dim == ((len(self.states),) * 2)
        self.generator = generator

    def set_timestep(self, dt):
        assert isinstance(dt, float) and dt < 1.0
        self.dt = dt

    def step(self, x):
        assert self.dt is not None, "Time step was not set!"

        prob = expm(self.dt * self.generator.apply(x))
        prob = np.cumsum(prob[self.current_state, :])

        xi = np.random.random()
        if xi < prob[0]:
            self.current_state = 0
        elif prob[-2] <= xi:
            self.current_state = self.states[-1]
        else:
            for i in range(1, len(self.states)-1):
                if prob[i-1] <= xi < prob[i]:
                    self.current_state = i
                    break

        return self.current_state


class WienerProcess(object):
    
    def __init__(self, dim=1, dt=None):
        assert isinstance(dim, int) and dim >= 1, "dim must be a positive integer"
        assert (dt is None) or (isinstance(dt, (float, int)) and dt > 0), "dt must be either None or a positive number"

        self.dim = dim
        if dt is None:
            self.dt = None
        else:
            self.dt = float(dt)

    def step(self, dt=None):
        if self.dt is None:
            assert dt is not None, "Instance was created without fixed step size, so dt must not be None!"
            assert isinstance(dt, (float, int)) and dt > 0, "dt must be a positive number"
            dt = float(dt)
        else:
            if dt is not None:
                warnings.warn("Instance was created with fixed step size, so the parameter dt will be ignored!")
            dt = self.dt

        if self.dim == 1:
            return np.random.normal(0.0, np.sqrt(dt))
        else:
            return np.random.multivariate_normal(np.zeros((self.dim,)), np.eye(self.dim) * np.sqrt(dt))