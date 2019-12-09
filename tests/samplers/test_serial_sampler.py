import unittest

import gym

from muzero.samplers.serial_sampler import SerialSampler, rollout


class MockAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def step(self, *args, **kwargs):
        return self.action_space.sample(), {}


class MockEnv:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(5)

    def reset(self):
        self.num_steps = 0
        return self.observation_space.sample()

    def step(self, *args, **kwargs):
        self.num_steps += 1
        terminal = self.num_steps >= self.max_steps
        return self.observation_space.sample(), 0.0, terminal, {}


class TestRollout(unittest.TestCase):
    def test_rollout(self):
        terminal_steps = 5
        env = MockEnv(max_steps=terminal_steps)
        agent = MockAgent(env.action_space)

        # Greater-than-terminal case.
        traj = rollout(env, agent, max_steps=10)
        self.assertEqual(len(traj["s"]), terminal_steps + 1)
        self.assertEqual(len(traj["a"]), terminal_steps)
        self.assertEqual(len(traj["r"]), terminal_steps)
        self.assertEqual(len(traj["t"]), terminal_steps)
        self.assertEqual(len(traj["a_info"]), terminal_steps + 1)
        self.assertTrue(all(isinstance(v, dict) for v in traj["a_info"]))

        # Less-than-terminal case.
        traj = rollout(env, agent, max_steps=1)
        self.assertEqual(len(traj["s"]), 2)
        self.assertEqual(len(traj["a"]), 1)
        self.assertEqual(len(traj["r"]), 1)
        self.assertEqual(len(traj["t"]), 1)
        self.assertEqual(len(traj["a_info"]), 2)


class TestSerialSampler(unittest.TestCase):
    def test_sample(self):
        env = MockEnv(max_steps=4)
        sampler = SerialSampler(env, max_steps=10, max_rollout_steps=10000)

        agent = MockAgent(env.action_space)
        trajs = sampler.sample(agent)
        self.assertEqual(len(trajs), 3)
        self.assertEqual(sum(len(traj["a"]) for traj in trajs), 10)

        env = MockEnv(max_steps=5)
        sampler = SerialSampler(env, max_steps=10, max_rollout_steps=10000)
        trajs = sampler.sample(agent)
        self.assertEqual(len(trajs), 2)
        self.assertEqual(sum(len(traj["a"]) for traj in trajs), 10)


if __name__ == '__main__':
    unittest.main()
