# -*- coding: utf-8 -*-

import numpy as np

from la import OperatorInterface
from stochasticprocesses import DiscreteTimeMarkovChain


class ProblemDescription(object):

    def __init__(self, drift, diffusion, markov_chain, initial_condition):
        assert isinstance(drift, OperatorInterface)
        assert isinstance(diffusion, OperatorInterface)
        assert isinstance(markov_chain, DiscreteTimeMarkovChain)
        assert isinstance(initial_condition, np.ndarray)

        self.drift = drift
        self.diffusion = diffusion
        self.markov_chain = markov_chain
        self.initial_condition = initial_condition
