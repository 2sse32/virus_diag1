"""Couche d'accès simple : transforme un dictionnaire patient en diagnostic."""
from app.interface_clinique import ClinicalPredictor
from core.dataset import FEATURES


def diagnose_patient(model, scaler, patient: dict):
    predictor = ClinicalPredictor(model, scaler)
    values = [patient[f] for f in FEATURES]
    return predictor.diagnose(values)
