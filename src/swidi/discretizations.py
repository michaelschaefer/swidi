# -*- coding: utf-8 -*-


from __future__ import division

import numpy as np

from interfaces import DiscretizationInterface
from problem import ProblemDescription
from stochasticprocesses import WienerProcess


class EulerMaruyamaDiscretization(DiscretizationInterface):

    def __init__(self, problem, time_range=(0, 1), time_intervals=100):
        assert isinstance(problem, ProblemDescription)
        assert isinstance(time_range, tuple), "time_range must be a tuple"
        assert all([isinstance(t, (float, int)) for t in time_range]), "time_range must be a tuple of integers"
        assert len(time_range) == 2 and time_range[0] < time_range[1], "time_range must be a tuple of two increasing " \
                                                                       "integer numbers"
        assert isinstance(time_intervals, int) and time_intervals > 0, "time_intervals must be a positive integer"

        super(EulerMaruyamaDiscretization, self).__init__()
        self._problem = problem
        self._time_range = time_range
        self._time_intervals = time_intervals
        self._time_steps = np.linspace(time_range[0], time_range[1], time_intervals+1)
        self.dt = (self._time_range[1] - self._time_range[0]) / time_intervals
        self.problem.markov_chain.dt = self.dt

    def solve(self, return_state=False):
        diffusion = self.problem.diffusion
        drift = self.problem.drift
        initial_condition = self.problem.initial_condition
        markov_chain = self.problem.markov_chain

        dim = self.problem.diffusion.range.dim
        n = dim[0]
        m = 1 if len(dim) == 1 else dim[1]

        dw = WienerProcess(m, self.dt)

        states = np.zeros((self.time_intervals+1,))
        trajectory = np.zeros((self.time_intervals+1, n))

        for i, t in enumerate(self._time_steps):
            if i == 0:
                r = markov_chain.state()
                x = initial_condition
            else:
                r = markov_chain.evolve(x)
                x += drift.evaluate(x, r, t) * self.dt + (diffusion.evaluate(x, r, t).dot(dw.step())).ravel()

            states[i] = r
            trajectory[i, :] = x.copy()

        if return_state is False:
            return trajectory
        else:
            return trajectory, states
