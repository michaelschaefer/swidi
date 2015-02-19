# -*- coding: utf-8 -*-


from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np


class OperatorInterface(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self._range = None
        self._source = None

    @abstractmethod
    def apply(self, u, r=None):
        pass

    @property
    def range(self):
        return self._range

    @property
    def source(self):
        return self._source


class VectorSpace(object):

    def __init__(self, dim):
        assert isinstance(dim, tuple) and all([isinstance(d, int) for d in dim])
        self.dim = dim

    def ones(self):
        return np.ones(self.dim)

    def zeros(self):
        return np.zeros(self.dim)
