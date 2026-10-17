import numpy as np


class StandardScaler:
    """Centre et réduit chaque colonne."""

    def fit(self, X):
        self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (np.atleast_2d(X) - self.mean) / self.std
