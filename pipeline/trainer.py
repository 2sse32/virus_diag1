from utils.preprocessing import StandardScaler


class Trainer:
    """Responsable UNIQUEMENT de l'entraînement."""

    def __init__(self, model):
        self.model = model
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X = self.scaler.fit(X_train).transform(X_train)
        self.model.fit(X, y_train)
        return self.model, self.scaler
