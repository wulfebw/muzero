import collections

import numpy as np
import scipy.stats

import muzero.planning.spaces_util as spaces_util


class TabularModel:
    def __init__(self, observation_space, action_space, value_lr=0.1):
        self.s2i = spaces_util.get_space_to_index_converter(observation_space)
        self.a2i = spaces_util.get_space_to_index_converter(action_space)

        self.t = dict()
        self.v = collections.defaultdict(float)
        self.pi = collections.defaultdict(lambda: np.ones(action_space.n))

    def update(self, samples):
        for (s, a, r, sp, t, g) in samples:
            si = self.s2i(s)
            ai = self.a2i(a)
            self.t[(si, ai)] = (sp, r, t)
            self.v[si] -= self.value_lr * (self.v[si] - g)
            self.pi[si][ai] += 1

    def represent(self, s):
        return self.s2i(s)

    def transition(self, s, a):
        si = self.s2i(s)
        ai = self.a2i(a)
        if (si, ai) not in self.t:
            # If we've never seen this transition, make something up.
            return (si, 0.0, False)
        return self.t[(si, ai)]

    def predict(self, s):
        si = self.s2i(s)
        return scipy.stats.dirichlet.mean(self.pi[si]), self.v[si]
