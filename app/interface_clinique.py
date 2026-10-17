"""Exercice : classe ClinicalPredictor (prédiction seule, sans entraînement)."""


class ClinicalPredictor:
    def __init__(self, model, scaler=None):
        # 1) reçoit un modèle DÉJÀ entraîné (il suffit qu'il ait predict())
        self.model = model
        self.scaler = scaler

    def diagnose(self, patient_data):
        # 2) méthode diagnose(patient_data)
        x = self.scaler.transform(patient_data) if self.scaler else patient_data
        proba = float(self.model.predict(x)[0])
        # 3) Infecté si prédiction >= 0.5, sinon Sain
        return "Infecté" if proba >= 0.5 else "Sain"

    def probability(self, patient_data):
        """Probabilité d'infection (entre 0 et 1), utile pour l'interface web."""
        x = self.scaler.transform(patient_data) if self.scaler else patient_data
        return float(self.model.predict(x)[0])
