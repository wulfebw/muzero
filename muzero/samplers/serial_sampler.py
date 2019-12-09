import collections


def rollout(env, agent, max_steps):
    """Collects a single rollout of experience.

    Args:
        env: The environment to interact with (adheres to gym interface).
        agent: The agent acting in the environment.
        max_steps: The max number of steps to take in the environment.

    Returns:
        A dictionary of lists containing information from the trajectory.
    """
    assert max_steps > 0
    traj = collections.defaultdict(list)

    def add_step(**kwargs):
        for k, v in kwargs.items():
            traj[k].append(v)

    s = env.reset()
    num_steps = 0
    while num_steps < max_steps:
        a, a_info = agent.step(s)
        sp, r, t, _ = env.step(a)
        add_step(s=s, a=a, r=r, t=t, a_info=a_info)

        s = sp
        num_steps += 1

        if t:
            break

    # Handle certain edge cases during sampling.
    # 1. Ensure there's always a next state.
    traj["s"].append(s)
    # 2. Ensure that the agent info (importantly containing the next-state-value) always exists.
    _, a_info = agent.step(s)
    traj["a_info"].append(a_info)

    return traj


class SerialSampler:
    """A sampler that serially samples trajectories from an environment."""

    def __init__(self, env, max_steps, max_rollout_steps=1e10):
        """
        Args:
            env: The environment to sample from.
            max_steps: The max steps to take total per sample call.
            max_rollout_steps: The max steps to take within a single rollout.
        """
        self.env = env
        self.max_steps = max_steps
        self.max_rollout_steps = max_rollout_steps

    def sample(self, agent):
        trajs = []
        num_steps = 0
        while num_steps < self.max_steps:
            steps_left = self.max_steps - num_steps
            traj = rollout(self.env, agent, min(self.max_rollout_steps, steps_left))
            trajs.append(traj)
            num_steps += len(traj["a"])
        return trajs
