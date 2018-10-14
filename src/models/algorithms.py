from ..utils import logger

from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def available_algorithms(keys_only=True):
    """Valid Algorithms for training or prediction

    This function simply returns a dict of known
    algorithms strings and their corresponding estimator function.

    It exists to allow for a description of the mapping for
    each of the valid strings as a docstring

    The valid algorithm names, and the function they map to, are:


    ============                 ====================================
    Algorithm                    Function
    ============                 ====================================
    LinearSVC                    sklearn.svm.LinearSVC
    GradientBoostingClassifier   sklearn.ensemble.GradientBoostingClassifier
    ============                 ====================================

    Parameters
    ----------
    keys_only: boolean
        If True, return only keys. Otherwise, return a dictionary mapping keys to algorithms
    """
    if keys_only:
        return list(_ALGORITHMS.keys())
    return _ALGORITHMS


class ComboGridSearchCV(BaseEstimator):
    '''
    Parameters
    ----------
    alg_name: str
        name of the algorithm to perform grid search over. name should
        be a key in the available_algorithms
    alg_params: dict
        these are the parameters for alg_name that will be fixed
    gridsearch_params: dict
        dict of parameters to grid search over
    params: dict
        other parameters to GridSearchCV
    '''
    def __init__(self, gridsearch_params=None,
                 alg_name=None,
                 alg_params=None,
                 params=None):
        '''
        Init a GridSearchCV with the appropriate input parameters.
        '''
        self.gridsearch_params = gridsearch_params
        self.alg_name = alg_name
        self.alg_params = alg_params
        self.params = params
        self.GSCV_ = None

    def fit(self, X, y=None, **kwargs):
        alg = available_algorithms()[self.alg_name]
        alg.set_params(**self.alg_params)
        self.GSCV_ = GridSearchCV(alg, self.gridsearch_params, **self.params)
        self.GSCV_.fit(X, y=y, **kwargs)

    def transform(self, X, y=None, **kwargs):
        if self.GSCV_ is None:
            logger.warning("fit must be run before transform")
        return self.GSCV_.transform(X, y=y, **kwargs)

    def predict(self, X, **kwargs):
        if self.GSCV_ is None:
            logger.warning("fit must be run before precit")
        return self.GSCV_.predict(X, **kwargs)


_ALGORITHMS = {
    'linearSVC': LinearSVC(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'GridSearchCV': ComboGridSearchCV(),
    'RandomForestClassifier': RandomForestClassifier(),
}
