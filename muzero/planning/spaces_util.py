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

        def _converter(index):
            return np.unravel_index(index, space.nvec)

        return _converter
    else:
        raise _make_not_implemented_error(space)


def get_space_to_index_converter(space):
    if isinstance(space, gym.spaces.Discrete):
        return lambda x: x
    elif isinstance(space, gym.spaces.MultiDiscrete):

        def _converter(element):
            return np.ravel_multi_index(element, space.nvec)

        return _converter

    else:
        raise _make_not_implemented_error(space)
