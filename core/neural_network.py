import numpy as np
from core.model import Model
from core.optimizer import SGD
from core.logistic_regression import sigmoid


class NeuralNetwork(Model):
    """Réseau à 1 couche cachée (tanh) et sortie sigmoïde."""

    def __init__(self, hidden=8, optimizer=None, epochs=500, seed=0):
        self.hidden, self.epochs, self.seed = hidden, epochs, seed
        self.optimizer = optimizer or SGD(lr=0.1)

    def fit(self, X, y):
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, 0.5, (d, self.hidden)); self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, 0.5, self.hidden);      self.b2 = np.zeros(1)
        for _ in range(self.epochs):
            h = np.tanh(X @ self.W1 + self.b1)
            p = sigmoid(h @ self.W2 + self.b2)
            dz2 = (p - y) / n
            dW2, db2 = h.T @ dz2, np.array([dz2.sum()])
            dh = np.outer(dz2, self.W2) * (1 - h ** 2)
            dW1, db1 = X.T @ dh, dh.sum(axis=0)
            self.optimizer.update([self.W1, self.b1, self.W2, self.b2],
                                  [dW1, db1, dW2, db2])
        return self

    def predict(self, X):
        h = np.tanh(np.atleast_2d(X) @ self.W1 + self.b1)
        return sigmoid(h @ self.W2 + self.b2)
