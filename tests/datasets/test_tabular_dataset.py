import unittest

from muzero.datasets.tabular_dataset import TabularDataset


class TestTabularDataset(unittest.TestCase):
    def test_traj_to_samples(self):
        s = [0, 1, 2, 3]
        a = [4, 5, 6]
        r = [10, 11, 12]
        v = [1, 2, 3, -1]
        a_info = [dict(v=vi) for vi in v]
        discount = 0.9

        traj = dict(
            s=s,
            a=a,
            r=r,
            t=[0, 0, 1],
            a_info=a_info,
        )
        dset = TabularDataset(1, discount)
        samples = dset._traj_to_samples(traj)
        g0 = r[0] + discount * v[1]
        self.assertEqual(samples[0], (s[0], a[0], r[0], 1, 0, g0))
        g1 = r[1] + discount * v[2]
        self.assertEqual(samples[1], (s[1], a[1], r[1], 2, 0, g1))
        g2 = r[2]
        self.assertEqual(samples[2], (s[2], a[2], r[2], 3, 1, g2))

        traj = dict(
            s=s,
            a=a,
            r=r,
            t=[0, 0, 0],
            sp=[1, 2, 3],
            a_info=a_info,
        )
        dset = TabularDataset(2, discount)
        samples = dset._traj_to_samples(traj)
        g0 = r[0] + discount * r[1] + discount**2 * v[2]
        self.assertEqual(samples[0][-1], g0)
        g1 = r[1] + discount * r[2] + discount**2 * v[3]
        self.assertEqual(samples[1][-1], g1)
        g2 = r[2] + discount * v[3]
        self.assertEqual(samples[2][-1], g2)


if __name__ == '__main__':
    unittest.main()
