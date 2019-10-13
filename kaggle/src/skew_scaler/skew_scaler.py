import numpy as np
from scipy.stats import skewnorm


class SkewScaler():
    def __init__(self,):
        self._a = None
        self._loc = None
        self._scale = None

    def fit(self, x, y=None):
        x = np.asarray(x)
        assert x.ndim == 1
        self._a, self._loc, self._scale = skewnorm.fit(x)
        return self

    def transform(self, x):
        q = skewnorm.cdf(x=x, a=self._a, loc=self._loc, scale=self._scale)
        z = skewnorm.ppf(q=q, a=0, loc=0, scale=1)
        return z

    def inverse_transform(self, z, copy=None):
        q = skewnorm.cdf(x=z, a=0, loc=0, scale=1)
        x = skewnorm.ppf(q=q, a=self._a, loc=self._loc, scale=self._scale)
        return x

    def cdf(self, x):
        q = skewnorm.cdf(x=x, a=self._a, loc=self._loc, scale=self._scale)
        return q
