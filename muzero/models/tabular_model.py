import collections

import numpy as np
import scipy.stats
import muzero.planning.spaces_util as spaces_util


def _assert_is_index(v):
    assert isinstance(v, (int, np.integer)), "Value should be index, but got value: {} of type: {}".format(
        v, type(v))


class TabularModel:
    def __init__(self, observation_space, action_space, value_lr=0.05):
        """
        Args:
            observation_space: The observation space of the environment.
            action_space: The action space of the environment.
            value_lr: The learning rate to use in updating the value of a state.
        """
        # These functions convert from the observation / action space to an index
        # in a table (represented as a dictionary) for tracking model parameters.
        self.s2i = spaces_util.get_space_to_index_converter(observation_space)
        self.a2i = spaces_util.get_space_to_index_converter(action_space)

        # The transition function mapping from (state, action) pairs to
        # a tuple of (next_state, reward, terminal).
        self.t = dict()
        # The state-value function.
        self.v = collections.defaultdict(float)
        # The policy, which we represent as a dirichlet distribution with
        # a prior of ones for each action.
        self.pi = collections.defaultdict(lambda: np.ones(action_space.n))

        self.value_lr = value_lr

        # These are for debugging purposes.
        self._i2s = spaces_util.get_index_to_space_converter(observation_space)
        self._i2a = spaces_util.get_index_to_space_converter(action_space)

    def update(self, samples):
        """Update the parameters of this model using the samples (i.e., learning).

        Args:
            samples: A list of tuples. Each tuple contains (state, action, reward,
                next_state, terminal, return).
        """
        for (s, a, r, sp, t, g) in samples:
            si = self.s2i(s)
            ai = self.a2i(a)
            spi = self.s2i(sp)
            # "Learning" of the transition in the deterministic case just notes the transition.
            self.t[(si, ai)] = (spi, r, t)
            # Incremental estimation of the value function.
            self.v[si] -= self.value_lr * (self.v[si] - g)
            # Each time we take an action in a state, we increment the dirichlet prior count.
            self.pi[si][ai] += 1

        pi, v = self.env_pi_v()
        return dict(pi=pi, v=v)

    def represent(self, s):
        """Represent a state with an internal representation.

        This corresponds to the representation function, `h` in the paper.
        In this case, we just return an index of the state.

        Args:
            s: The state to represent.

        Returns:
            The representation of the state.
        """
        return self.s2i(s)

    def transition(self, s, a):
        """Provides the transition function of the model.

        This corresponds to the transition function, `g(s^t-1, a^t) -> (s^t, r^t)`.
        We also return "terminal" (whether (s, a, s') is a terminal transition).

        Args:
            s: The state (in index format) from which to transition.
            a: The action taken in that state.
        
        Returns:
            Tuple of (next_state, reward, terminal).
        """
        _assert_is_index(s)
        ai = self.a2i(a)
        if (s, ai) not in self.t:
            # If we've never seen this transition, make something up.
            # The only important thing is that this be terminal.
            return (s, 0.0, True)
        return self.t[(s, ai)]

    def predict(self, s):
        """Predicts the policy and value function of a state.

        This corresponds to the prediction function `f(s) -> (pi, v)`.

        Args:
            s: The state (in index format) for which to predict.

        Returns:
            A tuple of a policy (probability distribution over actions
            for this state), and the value of the state.
        """
        _assert_is_index(s)
        probs = scipy.stats.dirichlet.mean(self.pi[s])
        pi = {i: p for (i, p) in enumerate(probs)}
        return pi, self.v[s]

    def env_pi_v(self):
        """Returns the policy and value function in environment format."""
        v = dict()
        pi = dict()
        for (si, vi) in self.v.items():
            s = self._i2s(si)
            v[s] = vi
            pi[s] = np.argmax(self.pi[si])
        return pi, v
