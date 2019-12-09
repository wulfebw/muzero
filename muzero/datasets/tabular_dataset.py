class TabularDataset:
    """Converts trajectories into samples.

    This class is effectively const, and exists as an object
    to store the n-step return value and the discount, and
    because in the function-approximation case this class is more complicated.
    """

    def __init__(self, return_n, discount):
        self.return_n = return_n
        self.discount = discount

    def trajectories_to_samples(self, trajectories):
        samples = []
        for traj in trajectories:
            samples.extend(self._traj_to_samples(traj))
        return samples, self._get_info(samples)

    def _traj_to_samples(self, traj):
        length = len(traj["a"])
        samples = []
        for i in range(length):
            # O(m * n) return calculation even though O(m) is possible :(
            g = 0
            for j in range(self.return_n):
                g += self.discount**j * traj["r"][i + j]
                if i + j + 1 == length:
                    break
            # This `i + j + 1` index into `v` only works because during
            # sampling we sample an extra value for `s'` at the end of the trajectory.
            # Note that v[k] corresponds to the value of s[k], and we want to take the
            # value of s'[k] so we have to consider v[k + 1].
            # Also, `terminal[i + j]` indicates whether (s, a, s') is a terminal transition.
            g += self.discount**(j + 1) * traj["a_info"][i + j + 1]["v"] * (0 if traj["t"][i + j] else 1)
            samples.append((
                traj["s"][i],
                traj["a"][i],
                traj["r"][i],
                traj["s"][i + 1],
                traj["t"][i],
                g,
            ))
        return samples

    def _get_info(self, samples):
        avg_return = sum(s[-1] for s in samples) / (len(samples) + 1e-8)
        return dict(avg_discounted_return=avg_return)
