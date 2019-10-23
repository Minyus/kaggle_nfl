import numpy as np
from scipy.stats import lognorm


class LogNormScaler:
    def __init__(self,):
        self._loc = None
        self._scale = None

    def fit(self, x, y=None):
        x = np.asarray(x)
        assert x.ndim == 1
        self._loc, self._scale = lognorm.fit(x)
        return self

    def transform(self, x):
        q = lognorm.cdf(x=x, loc=self._loc, scale=self._scale)
        z = lognorm.ppf(q=q, loc=0, scale=1)
        return z

    def inverse_transform(self, z, copy=None):
        q = lognorm.cdf(x=z, loc=0, scale=1)
        x = lognorm.ppf(q=q, loc=self._loc, scale=self._scale)
        return x

    def cdf(self, x):
        q = lognorm.cdf(x=x, loc=self._loc, scale=self._scale)
        return q
