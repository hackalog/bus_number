import sklearn

_ALGORITHMS = {
}

def available_algorithms():
    """Valid Algorithms for training or prediction

    This function simply returns a dict of known
    algorithms strings and their corresponding estimator function.

    It exists to allow for a description of the mapping for
    each of the valid strings as a docstring

    The valid algorithm names, and the function they map to, are:

    ============     ====================================
    Algorithm        Function
    ============     ====================================

    ============     ====================================
    """
    return _ALGORITHMS

