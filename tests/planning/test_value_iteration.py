import unittest

import numpy as np

from muzero.envs.line import Line
from muzero.envs.maze import Maze
from muzero.planning.value_iteration import value_iteration


class TestValueIteration(unittest.TestCase):
    def test_on_line_env(self):
        env = Line()
        v, pi, _, _ = value_iteration(env, max_iter=env.length, verbose=False)
        np.testing.assert_array_almost_equal(v, [6, 7, 8, 9, 10])
        np.testing.assert_array_almost_equal(pi, np.zeros_like(pi))

    def test_on_maze_env(self):
        env = Maze()
        _, _, v, pi = value_iteration(env, max_iter=100, verbose=False)
        self.assertAlmostEqual(v[(4, 4)], 1)
        self.assertAlmostEqual(v[(5, 0)], -1)
        self.assertAlmostEqual(v[(0, 0)], env.discount**8, 6)


if __name__ == '__main__':
    unittest.main()
