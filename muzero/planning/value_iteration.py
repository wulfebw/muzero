import numpy as np

import muzero.planning.spaces_util as spaces_util


def value_iteration(mdp, max_iter, tol=1e-4, verbose=True):
    num_states = spaces_util.get_space_size(mdp.observation_space)
    i2s = spaces_util.get_index_to_space_converter(mdp.observation_space)
    s2i = spaces_util.get_space_to_index_converter(mdp.observation_space)
    num_actions = spaces_util.get_space_size(mdp.action_space)
    i2a = spaces_util.get_index_to_space_converter(mdp.action_space)

    v = np.zeros(num_states)
    pi = np.empty(num_states, dtype=int)
    for itr in range(max_iter):
        residual = 0
        for s in range(num_states):
            max_value = -np.inf
            best_a = 0
            for a in range(num_actions):
                cur_value = 0
                for sp, r, t, p in mdp.transitions(i2s(s), i2a(a)):
                    cur_value += p * r
                    if not t:
                        cur_value += p * mdp.discount * v[s2i(sp)]
                if cur_value > max_value:
                    max_value = cur_value
                    best_a = a
            assert max_value != -np.inf, "Max value was not set"
            residual = max(residual, abs(v[s] - max_value))
            v[s] = max_value
            pi[s] = best_a
        if verbose:
            print("itr: {} / {} residual: {}".format(itr + 1, max_iter, residual))
        if residual < tol:
            break

    # Convert to a format reconginzable by the mdp.
    v_dict = dict()
    pi_dict = dict()
    for s in range(num_states):
        v_dict[i2s(s)] = v[s]
        pi_dict[i2s(s)] = pi[s]
    return v, pi, v_dict, pi_dict
