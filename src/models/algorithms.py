from sklearn.svm import LinearSVC

_ALGORITHMS = {
    'linearSVC': LinearSVC()
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
    LinearSVC        sklearn.svm.LinearSVC
    ============     ====================================
    """
    return _ALGORITHMS

