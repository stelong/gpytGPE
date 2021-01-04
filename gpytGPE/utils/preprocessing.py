import numpy as np


class Scaler:
    def __init__(self):
        self.mu = []
        self.sigma = []

    def fit(self, X):
        self.in_dim = X.shape[1]
        for i in range(self.in_dim):
            self.mu.append(np.mean(X[:, i]))
            self.sigma.append(np.std(X[:, i]))

    def transform(self, X):
        X_ = np.full_like(X, 0, dtype=float)
        for i in range(self.in_dim):
            X_[:, i] = (X[:, i] - self.mu[i]) / self.sigma[i]
        return X_

    def inverse_transform(self, X):
        X_ = np.full_like(X, 0, dtype=float)
        for i in range(self.in_dim):
            X_[:, i] = self.sigma[i] * X[:, i] + self.mu[i]
        return X_

    def inverse_transform_predictions(self, mean, std):
        return self.sigma[0] * mean + self.mu[0], self.sigma[0] * std
