# -*- coding: utf-8 -*-

import numpy as np

from swidi.discretizations import EulerMaruyamaDiscretization
from swidi.interfaces import FunctionInterface, VectorSpace
from swidi.problem import ProblemDescription
from swidi.stochasticprocesses import DiscreteTimeMarkovChain
from swidi.visualization import visualize_trajectory_1d


class DiffusionOperator(FunctionInterface):

    def __init__(self):
        super(DiffusionOperator, self).__init__()
        self._source = VectorSpace((1,))
        self._range = VectorSpace((1, 1))

    def evaluate(self, x, r=None, t=None):
        assert r is None or isinstance(r, (float, int))
        if r is None:
            return self._range.ones()
        else:
            return self._range.ones() * r


class DriftOperator(FunctionInterface):

    def __init__(self):
        super(DriftOperator, self).__init__()
        self._source = VectorSpace((1,))
        self._range = VectorSpace((1,))

    def evaluate(self, x, r, t):
        return 1.0


class MarkovChainGenerator(FunctionInterface):

    def __init__(self):
        super(MarkovChainGenerator, self).__init__()
        self._dim = 2
        self._source = VectorSpace((self._dim,))
        self._range = VectorSpace((self._dim,)*2)

    def evaluate(self, x, r=None, t=None):
        return np.array([[-10, 10], [10, -10]])


markov_generator = MarkovChainGenerator()
markov_chain = DiscreteTimeMarkovChain([0, 1], markov_generator, initial_state=0)

diffusion = DiffusionOperator()
drift = DriftOperator()
initial_condition = np.array([0.0])
problem = ProblemDescription(drift, diffusion, markov_chain, initial_condition)

discretization = EulerMaruyamaDiscretization(problem, time_range=(0, 1), time_intervals=1000)
trajectory, states = discretization.solve(return_state=True)
visualize_trajectory_1d(trajectory, states=states, discretization=discretization)
