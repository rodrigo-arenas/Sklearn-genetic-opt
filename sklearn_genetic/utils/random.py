import random


def weighted_choice(weight):
    """
    Parameters
    ----------
    weight: float
        Weight of choosing a chromosome

    Returns
    -------
        Bool random (not uniform) choice
    """

    # This help to don't generate individuals of the same size on average
    p = random.uniform(0, weight)
    choice = random.choices([0, 1], [1-p, p])[0]

    return choice
