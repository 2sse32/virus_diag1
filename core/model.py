"""Classe de base (interface) commune à tous les modèles."""
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Entraîne le modèle."""

    @abstractmethod
    def predict(self, X):
        """Retourne une probabilité d'infection entre 0 et 1."""
