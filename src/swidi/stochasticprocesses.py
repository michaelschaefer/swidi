# -*- coding: utf-8 -*-


import numpy as np
from scipy.linalg import expm
import warnings

from interfaces import FunctionInterface, MarkovChainInterface


class DiscreteTimeMarkovChain(MarkovChainInterface):

    def __init__(self, states, generator, initial_state=None, dt=None):
        assert isinstance(states, (list, tuple)) and len(states) > 0
        assert isinstance(generator, FunctionInterface)
        assert initial_state is None or initial_state in states
        assert dt is None or (isinstance(dt, float) and dt > 0)

        super(MarkovChainInterface, self).__init__()

        self._states = states
        self.number_of_states = len(self._states)
        self._dt = dt

        if initial_state is None:
            self._current_state = self._states[0]
        else:
            self._current_state = self._states.index(initial_state)

        assert generator.range.dim == ((self.number_of_states,) * 2)
        self.generator = generator

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        assert isinstance(value, float) and float > 0
        if self._dt is None:
            self._dt = value
        else:
            warnings.warn("This implementation allows the time step to be set only once!")

    def evolve(self, x, dt=None):
        assert self.dt is not None, "Time step was not set!"
        if dt is not None:
            warnings.warn("This implementation uses a fixed time step. Parameter dt will be ignored!")

        prob = expm(self.dt * self.generator.evaluate(x))
        prob = np.cumsum(prob[self._current_state, :])

        xi = np.random.random()
        if xi < prob[0]:
            self._current_state = 0
        elif prob[-2] <= xi:
            self._current_state = self.number_of_states - 1
        else:
            for i in range(1, self.number_of_states - 1):
                if prob[i-1] <= xi < prob[i]:
                    self._current_state = i
                    break

        return self._states[self._current_state]

    def reset(self, initial_state=None):
        assert initial_state is None or initial_state in self._states
        self._current_state = initial_state

    def state(self):
        return self._states[self._current_state]


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