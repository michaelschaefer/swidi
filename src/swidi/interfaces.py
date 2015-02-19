# -*- coding: utf-8 -*-


from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np


class DiscretizationInterface(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self._problem = None
        self._time_range = None
        self._time_intervals = None
        self._time_steps = None

    @abstractmethod
    def solve(self, return_state=False):
        pass

    @property
    def problem(self):
        return self._problem

    @property
    def time_range(self):
        return self._time_range

    @property
    def time_intervals(self):
        return self._time_intervals

    @property
    def time_steps(self):
        return self._time_steps


class FunctionInterface(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self._range = None
        self._source = None

    @abstractmethod
    def evaluate(self, x, r, t):
        pass

    @property
    def range(self):
        assert isinstance(self._range, VectorSpace)
        return self._range

    @property
    def source(self):
        assert isinstance(self._source, VectorSpace)
        return self._source


class MarkovChainInterface(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def evolve(self, x, dt=None):
        pass

    @abstractmethod
    def reset(self, initial_state=None):
        pass

    @abstractproperty
    def state(self):
        pass


class VectorSpace(object):

    def __init__(self, dim):
        assert isinstance(dim, tuple) and all([isinstance(d, int) for d in dim])
        self.dim = dim

    def ones(self):
        return np.ones(self.dim)

    def zeros(self):
        return np.zeros(self.dim)
