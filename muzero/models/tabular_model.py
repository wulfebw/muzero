import collections

import numpy as np
import scipy.stats
import muzero.planning.spaces_util as spaces_util


class TabularModel:
    def __init__(self, observation_space, action_space, value_lr=0.1):
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

    def update(self, samples):
        """Update the parameters of this model using the samples (i.e., learning).

        Args:
            samples: A list of tuples. Each tuple contains (state, action, reward,
                next_state, terminal, return).
        """
        for (s, a, r, sp, t, g) in samples:
            si = self.s2i(s)
            ai = self.a2i(a)
            # "Learning" of the transition in the deterministic case just notes the transition.
            self.t[(si, ai)] = (sp, r, t)
            # Estimate the value function by incremental estimation.
            self.v[si] -= self.value_lr * (self.v[si] - g)
            # Each time we take an action in a state, we increment the dirichlet prior count.
            self.pi[si][ai] += 1

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
            s: The state from which to transition.
            a: The action taken in that state.
        
        Returns:
            Tuple of (next_state, reward, terminal).
        """
        si = self.s2i(s)
        ai = self.a2i(a)
        if (si, ai) not in self.t:
            # If we've never seen this transition, make something up.
            return (si, 0.0, False)
        return self.t[(si, ai)]

    def predict(self, s):
        """Predicts the policy and value function of a state.

        This corresponds to the prediction function `f(s) -> (pi, v)`.

        Args:
            s: The state for which to predict.

        Returns:
            A tuple of a policy (probability distribution over actions
            for this state), and the value of the state.
        """
        si = self.s2i(s)
        return scipy.stats.dirichlet.mean(self.pi[si]), self.v[si]
