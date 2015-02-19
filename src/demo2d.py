# -*- coding: utf-8 -*-

import numpy as np

from swidi.discretizations import EulerMaruyamaDiscretization
from swidi.interfaces import FunctionInterface, VectorSpace
from swidi.problem import ProblemDescription
from swidi.stochasticprocesses import DiscreteTimeMarkovChain
from swidi.visualization import visualize_trajectory_2d


class DiffusionOperator(FunctionInterface):

    def __init__(self):
        super(DiffusionOperator, self).__init__()
        self._source = VectorSpace((2,))
        self._range = VectorSpace((2, 2))

    def evaluate(self, x, r=None, t=None):
        assert r is None or isinstance(r, (float, int))
        return self.range.ones() * 0.01 * (1 + r)


class DriftOperator(FunctionInterface):

    def __init__(self):
        super(DriftOperator, self).__init__()
        self._source = VectorSpace((2,))
        self._range = VectorSpace((2,))

    def evaluate(self, x, r, t):
        assert isinstance(r, int)
        if r == 0:
            a = np.array([[0, -1], [1, 0]])
        else:
            a = np.array([[-1, 2], [-2, -1]])
        return a.dot(x)


class MarkovChainGenerator(FunctionInterface):

    def __init__(self):
        super(MarkovChainGenerator, self).__init__()
        self._dim = 2
        self._source = VectorSpace((self._dim,))
        self._range = VectorSpace((self._dim,)*2)

    def evaluate(self, x, r=None, t=None):
        return np.array([[-1, 1], [2, -2]])


markov_generator = MarkovChainGenerator()
markov_chain = DiscreteTimeMarkovChain([0, 1], markov_generator, initial_state=0)

diffusion = DiffusionOperator()
drift = DriftOperator()
initial_condition = np.ones((2,))
problem = ProblemDescription(drift, diffusion, markov_chain, initial_condition)

discretization = EulerMaruyamaDiscretization(problem, time_range=(0, 10), time_intervals=1000)
trajectory = discretization.solve()
visualize_trajectory_2d(trajectory)
