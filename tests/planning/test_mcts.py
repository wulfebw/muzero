import unittest

from muzero.envs.line import Line
from muzero.planning.mcts import MCTS
from muzero.planning.spaces_util import get_space_size
from muzero.planning.value_iteration import value_iteration


class LineModel:
    """The "true" model for the Line environment used for testing."""
    def __init__(self):
        self.env = Line()
        _, _, v, pi = value_iteration(self.env, 10, verbose=False)
        self.v = v
        self.pi = pi

    def transition(self, s, a):
        transitions = self.env.transitions(s, a)
        assert len(transitions) == 1
        sp, r, t, _ = transitions[0]
        return sp, r, t

    @property
    def discount(self):
        return self.env.discount

    @property
    def default_prior(self):
        pi = dict()
        n = get_space_size(self.env.action_space)
        assert n > 0
        for a in range(n):
            pi[a] = 1 / n
        return pi

    def predict(self, state):
        pi = self.default_prior
        for k, v in pi.items():
            pi[k] = 0.0
        pi[self.pi[state]] = 1.0
        return pi, self.v[state]


class TestMCTS(unittest.TestCase):
    def test_on_line_env(self):
        model = LineModel()
        planner = MCTS(model, num_simulations=100, temp=1e-8)
        for state in range(model.env.length):
            pi, v = planner.plan(state, visualize=False)
            self.assertTrue(pi == {0: 1, 1: 0})
            self.assertAlmostEqual(v, model.v[state], 6)


if __name__ == '__main__':
    unittest.main()
