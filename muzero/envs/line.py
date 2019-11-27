import gym


class Line(gym.Env):
    """A simple environment for testing.

    - State space consists of cells on a line, where you start on the left and end on the right.
        State 0 is furthest to the left, state `length - 1` is furthest to the right.
    - Action space is move left / right.
    - Transition model is deterministic; stationary if walls prevent action.
    - Reward model is +10 for reaching the end, -1 for each action.
    - Discount factor is 1.0.
    """

    def __init__(self, length=5):
        self.length = length
        self.finish_reward = 10
        self.act_reward = -1
        self.observation_space = gym.spaces.Discrete(length)
        self.action_space = gym.spaces.Discrete(2)
        self.discount = 1.0

    def transitions(self, state, action):
        dx = 1 if action == 0 else -1
        next_state = min(max(state + dx, 0), self.length - 1)
        terminal = state == self.length - 1
        if terminal:
            reward = self.finish_reward if next_state == self.length - 1 else 0
        else:
            reward = self.act_reward
        prob = 1.0
        return [(next_state, reward, terminal, prob)]
