import gym
import numpy as np


def _make_not_implemented_error(space):
    return NotImplementedError("Space of type {} not implemented.".format(type(space)))


def get_space_size(space):
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return np.prod(space.nvec)
    else:
        raise _make_not_implemented_error(space)


def get_index_to_space_converter(space):
    if isinstance(space, gym.spaces.Discrete):
        return lambda x: x
    elif isinstance(space, gym.spaces.MultiDiscrete):
        index_sizes = tuple(space.nvec)
        return lambda x: np.unravel_index(x, index_sizes)
    else:
        raise _make_not_implemented_error(space)


def get_space_to_index_converter(space):
    if isinstance(space, gym.spaces.Discrete):
        return lambda x: x
    elif isinstance(space, gym.spaces.MultiDiscrete):
        index_sizes = tuple(space.nvec)
        return lambda x: np.ravel_multi_index(x, index_sizes)
    else:
        raise _make_not_implemented_error(space)
