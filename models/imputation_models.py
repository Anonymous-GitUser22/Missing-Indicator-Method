from sklearn.base import BaseEstimator, TransformerMixin
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.low_rank_gaussian_copula import LowRankGaussianCopula

class GCImputer(BaseEstimator, TransformerMixin):
    def __init__(self, **gc_params):
        # kwargs depend on the model used, so assign them whatever they are
        for key, value in gc_params.items():
            setattr(self, key, value)

        self._param_names = list(gc_params.keys())

    def get_params(self, deep=True):
        return {param: getattr(self, param)
                for param in self._param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y=None):
        gc_params = self.get_params()
        self.gc = GaussianCopula(**gc_params)
        self.gc.fit(X)
        return self.gc

    def transform(self, X, y=None):
        X_imputed = self.gc.transform(X)
        return X_imputed

class LRGCImputer(BaseEstimator, TransformerMixin):
    def __init__(self, **gc_params):
        # kwargs depend on the model used, so assign them whatever they are
        for key, value in gc_params.items():
            setattr(self, key, value)

        self._param_names = list(gc_params.keys())

    def get_params(self, deep=True):
        return {param: getattr(self, param)
                for param in self._param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y=None):
        gc_params = self.get_params()
        self.lrgc = LowRankGaussianCopula(**gc_params)
        self.lrgc.fit(X)
        return self.lrgc

    def transform(self, X, y=None):
        X_imputed = self.lrgc.transform(X)
        return X_imputed