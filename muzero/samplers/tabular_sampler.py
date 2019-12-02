import collections


def rollout(env, agent, max_steps):
    traj = collections.defaultdict(list)

    def add_step(**kwargs):
        for k, v in kwargs.items():
            traj[k].append(v)

    s = env.reset()
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
    traj["a_info"].append(s)

    return traj


class TabularSampler:
    def __init__(self, max_steps, max_rollout_steps):
        self.max_steps = max_steps
        self.max_rollout_steps = max_rollout_steps

    def sample(self, env, agent):
        trajs = []
        num_steps = 0
        while num_steps < max_steps:
            steps_left = max_steps - num_steps
            traj = rollout(env, agent, min(self.max_rollout_steps, steps_left))
            trajs.append(traj)
            num_steps += len(traj["a"])
        return trajs
