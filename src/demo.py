# -*- coding: utf-8 -*-

import numpy as np

from swidi.discretizations import EulerMaruyamaDiscretization
from swidi.interfaces import FunctionInterface, VectorSpace
from swidi.problem import ProblemDescription
from swidi.stochasticprocesses import DiscreteTimeMarkovChain
from swidi.visualization import visualize_trajectory_1d


class DiffusionOperator(FunctionInterface):

    def __init__(self, dim):
        super(DiffusionOperator, self).__init__()
        self._dim = dim
        self._source = VectorSpace((dim,))
        self._range = VectorSpace((dim,))

    def evaluate(self, x, r=None, t=None):
        assert r is None or isinstance(r, (float, int))
        if r is None:
            return self._range.ones()
        else:
            return self._range.ones() * r


class DriftOperator(FunctionInterface):

    def __init__(self, dim):
        super(DriftOperator, self).__init__()
        self._dim = dim
        self._source = VectorSpace((dim,))
        self._range = VectorSpace((dim,))

    def evaluate(self, x, r, t):
        return self._range.ones()


class MarkovChainGenerator(FunctionInterface):

    def __init__(self):
        super(MarkovChainGenerator, self).__init__()
        self._dim = 2
        self._source = VectorSpace((self._dim,))
        self._range = VectorSpace((self._dim,)*2)

    def evaluate(self, x, r=None, t=None):
        return np.array([[-10, 10], [10, -10]])


spacial_dimension = 1

markov_generator = MarkovChainGenerator()
markov_chain = DiscreteTimeMarkovChain([0, 1], markov_generator, initial_state=0)

diffusion = DiffusionOperator(spacial_dimension)
drift = DriftOperator(spacial_dimension)
initial_condition = np.zeros((spacial_dimension,))
problem = ProblemDescription(drift, diffusion, markov_chain, initial_condition)

discretization = EulerMaruyamaDiscretization(problem, time_range=(0, 1), time_intervals=1000)
trajectory, states = discretization.solve(return_state=True)
visualize_trajectory_1d(trajectory, states=states, discretization=discretization)
