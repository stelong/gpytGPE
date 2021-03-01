import numpy as np


class StandardScaler:
    def __init__(self, log_transform=False):
        self.log_transform = log_transform
        self.mu = None
        self.sigma = None

    def fit(self, X):
        if self.log_transform:
            X = np.log(X)
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)

    def transform(self, X):
        if self.log_transform:
            X = np.log(X)
        X_ = (X - self.mu) / self.sigma
        return X_

    def inverse_transform(self, X_):
        X = self.sigma * X_ + self.mu
        if self.log_transform:
            X = np.exp(X)
        return X

    def inverse_transform_predictions(self, y_mean_, y_std_):
        y_mean = self.sigma * y_mean_ + self.mu
        y_std = self.sigma * y_std_
        if self.log_transform:
            return (
                np.exp(y_mean + 0.5 * np.power(y_std, 2)),
                (np.exp(np.power(y_std, 2)) - 1)
                * np.exp(2 * y_mean + np.power(y_std, 2)),
            )
        else:
            return y_mean, y_std


class UnitCubeScaler:
    def __init__(self):
        self.a = None
        self.b = None

    def fit(self, X):
        self.a = np.min(X, axis=0)
        self.b = np.max(X, axis=0)

    def transform(self, X):
        X_ = (X - self.a) / (self.b - self.a)
        return X_

    def inverse_transform(self, X_):
        X = (self.b - self.a) * X_ + self.a
        return X
