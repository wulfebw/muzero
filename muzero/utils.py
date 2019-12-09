import numpy as np


def categorical_sample(d):
    """Randomly sample a value from a discrete set based on provided probabilities.
    
    Args:
        d: A dictionary mapping choices to probabilities.

    Returns:
        One of the possible choices.
    """
    choice_probs = list(d.items())
    probs = [t[1] for t in choice_probs]
    index = np.argmax(np.random.multinomial(1, probs))
    return choice_probs[index][0]
