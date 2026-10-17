from utils import metrics


class Evaluator:
    """Mesure la qualité d'un modèle déjà entraîné."""

    def __init__(self, model, scaler):
        self.model, self.scaler = model, scaler

    def evaluate(self, X_test, y_test):
        pred = (self.model.predict(self.scaler.transform(X_test)) >= 0.5).astype(int)
        return {
            "accuracy": metrics.accuracy(y_test, pred),
            "precision": metrics.precision(y_test, pred),
            "recall": metrics.recall(y_test, pred),
            "f1": metrics.f1(y_test, pred),
        }
