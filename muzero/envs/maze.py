import gym
import matplotlib.pyplot as plt
import numpy as np


class Maze(gym.Env):
    """A simple maze environment.
    _______________
    |      |   pit|
    |             |
    | _____|__  __|
    |             |
    |      |  end |
    |______|______|

    State space:
        - The current (x position, y position) of the agent (MultiDiscrete([6, 6])).
        - So the observation is the state.
        - Each room is 3 cells tall and wide, and there are four rooms.
        - The state (0, 0) is the upper-left-most cell.
        - The state (5, 5) is the bottom-right-most cell.
    
    Action space:
        - Move east, west, north or south (Discrete(4)).

    Transition model:
        - Determinstically move to the requested, adjacent cell if possible (given walls).
        - States in the pit are terminal. 
        - The pit is in the upper right portion of the upper right room.
        - States at the end location are terminal.
        - The end is at one cell off the bottom-most and one cell off the right-most cell.

    Reward model:
        - Exiting the maze is positive one (+1) reward.
        - Entering the pit is negative one (-1) reward.

    Discount factor:
        - The discount factor of the mdp is 0.99.
        - A discount factor is used instead of a negative reward for taking a step
            because the former tends to expose bugs in an algorithm.

    Initial state distribution:
        - The agent starts in a uniformly-randomly-selected cell in the top-left room.
        - The random initial state ensures the algorithm doesn't just work from a single initial state.

    Comments:
        - The shortest path depends on the initial state (through the bottom-left or upper-right rooms).
        - On average, the upper-right path is shorter, but since it has the pit an agent might
            learn not to enter that room; part of the point is to test for this behavior.
    """

    def __init__(self):
        self.length = 6
        self.height = 6

        self.start_states = [(i, j) for i in range(3) for j in range(3)]

        self.pit_state = (self.length - 1, 0)
        self.pit_reward = -1

        self.end_state = (self.length - 2, self.height - 2)
        self.end_reward = 1

        self.observation_space = gym.spaces.MultiDiscrete([self.length, self.height])
        self.action_space = gym.spaces.Discrete(4)

        self.wall_transitions = set([
            ((2, 0), (3, 0)),
            ((2, 2), (3, 2)),
            ((1, 2), (1, 3)),
            ((2, 2), (2, 3)),
            ((3, 2), (3, 3)),
            ((5, 2), (5, 3)),
            ((2, 4), (3, 4)),
            ((2, 5), (3, 5)),
        ])

        self.discount = 0.99
        self.state = None

    def reset(self):
        self.state = self.start_states[np.random.randint(len(self.start_states))]
        return self.state

    def _action_index_to_dx_dy(self, action):
        if action == 0:
            # East.
            return (1, 0)
        elif action == 1:
            # West.
            return (-1, 0)
        elif action == 2:
            # North. Note that up is decreasing in index.
            return (0, -1)
        else:
            # South. Note that down is increasing in index.
            return (0, 1)

    def _get_next_state(self, action):
        assert self.action_space.contains(action)

        if self.state == self.pit_state or self.state == self.end_state:
            return self.state

        dx, dy = self._action_index_to_dx_dy(action)
        x, y = self.state
        xp = min(max(x + dx, 0), self.length - 1)
        yp = min(max(y + dy, 0), self.height - 1)
        next_state = (xp, yp)
        invalid = ((self.state, next_state) in self.wall_transitions) or (
            (next_state, self.state) in self.wall_transitions)
        if invalid:
            return self.state
        return next_state

    def _get_reward_terminal(self, state):
        if state == self.pit_state:
            return self.pit_reward, True
        elif state == self.end_state:
            return self.end_reward, True
        return 0, False

    def step(self, action):
        # By computing the reward / terminal when actually in a terminal state,
        # we allow an agent to take an action in the terminal state. This
        # does not influence the policy, but it makes it easier to compare
        # value functions across algorithms.
        reward, terminal = self._get_reward_terminal(self.state)
        self.state = self._get_next_state(action)
        return self.state, reward, terminal, dict()

    def transitions(self, state, action):
        self.state = state
        next_state, reward, terminal, _ = self.step(action)
        return [(next_state, reward, terminal, 1.0)]

    def render(self):
        s = [["-" for _ in range(self.length)] for _ in range(self.height)]
        # The state is (x, y), but when visualizing x = col, and y = row.
        s[self.state[1]][self.state[0]] = "*"
        for row in s:
            print("".join(row))
            print()
        print()

    def render_value_function(self, v, mode="image", filepath=None):
        """Visualize a value function for this maze.

        Args:
            v: Dictionary mapping states to values.
            mode: The render mode; either "text" or "image".
            filepath: An optional filepath to direct output to if provided.
                This only works with the plot / image visualization.
        """
        viz = np.empty((self.height, self.length))
        for state, value in v.items():
            # The state is (x, y), but when visualizing x = col, and y = row.
            viz[state[1], state[0]] = value
        if mode == "text":
            print(viz)
        elif mode == "image":
            # Normalize to within colormap bounds.
            viz = (viz - np.min(viz)) / np.ptp(viz)
            plt.imshow(viz, cmap="jet")
            if filepath is not None:
                plt.savefig(filepath)
                plt.close()
            else:
                plt.show()
        else:
            raise NotImplementedError("Invalid mode: {}".format(mode))


if __name__ == "__main__":
    import time
    m = Maze()
    s = m.reset()
    while True:
        a = m.action_space.sample()
        ns, r, t, _ = m.step(a)
        m.render()
        if t:
            break
        time.sleep(0.05)
