# -*- coding: utf-8 -*-

import numpy as np

from interfaces import FunctionInterface
from stochasticprocesses import DiscreteTimeMarkovChain


class ProblemDescription(object):

    def __init__(self, drift, diffusion, markov_chain, initial_condition):
        assert isinstance(drift, FunctionInterface)
        assert isinstance(diffusion, FunctionInterface)
        assert drift.source.dim == diffusion.source.dim
        assert drift.range.dim[0] == diffusion.range.dim[0]
        assert isinstance(markov_chain, DiscreteTimeMarkovChain)
        assert isinstance(initial_condition, np.ndarray) and len(initial_condition.shape) == 1

        self.dim = initial_condition.shape[0]
        self.drift = drift
        self.diffusion = diffusion
        self.markov_chain = markov_chain
        self.initial_condition = initial_condition
