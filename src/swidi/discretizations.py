# -*- coding: utf-8 -*-


from __future__ import division

from abc import ABCMeta, abstractmethod
import numpy as np

from problem import ProblemDescription
from stochasticprocesses import WienerProcess


class DiscretizationInterface(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self._problem = None
        self._time_range = None
        self._time_intervals = None
        self._time_steps = None

    @abstractmethod
    def solve(self, return_states=False):
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
        self.problem.markov_chain.set_timestep(self.dt)

    def solve(self, return_states=False):
        diffusion = self.problem.diffusion
        drift = self.problem.drift
        initial_condition = self.problem.initial_condition
        markov_chain = self.problem.markov_chain

        dim = len(initial_condition)
        dw = WienerProcess(dim, self.dt)

        r = markov_chain.current_state
        x = initial_condition

        states = np.zeros((self.time_intervals+1,))
        states[0] = r
        trajectory = np.zeros((self.time_intervals+1, len(x)))
        trajectory[0, :] = x.copy()

        for k in range(self.time_intervals):
            r = markov_chain.step(x)
            x += drift.apply(x, r) * self.dt + diffusion.apply(x, r) * dw.step()

            states[k+1] = r
            trajectory[k+1, :] = x.copy()

        if return_states is False:
            return trajectory
        else:
            return trajectory, states
