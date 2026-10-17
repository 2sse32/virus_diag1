import numpy as np
from core.model import Model
from core.optimizer import SGD


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


class LogisticRegression(Model):
    def __init__(self, optimizer=None, epochs=500):
        self.optimizer = optimizer or SGD(lr=0.1)
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), np.zeros(1)
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            grad_w = X.T @ (p - y) / n
            grad_b = np.array([(p - y).mean()])
            self.optimizer.update([self.w, self.b], [grad_w, grad_b])
        return self

    def predict(self, X):
        return sigmoid(np.atleast_2d(X) @ self.w + self.b)
