"""Chargement, génération et découpage des données patients."""
import numpy as np
import pandas as pd

FEATURES = ["age", "temperature", "toux", "fatigue", "saturation_o2", "leucocytes", "crp"]
TARGET = "infecte"


def generate_patient_data(path="data/patient_data.csv", n=500, seed=42):
    """Génère un dataset synthétique de patients (infecté = 1, sain = 0)."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.45, n)  # ~45 % d'infectés
    inf = y == 1

    df = pd.DataFrame({
        "age": rng.integers(18, 85, n),
        "temperature": np.where(inf, rng.normal(38.6, 0.7, n), rng.normal(36.9, 0.4, n)).round(1),
        "toux": np.where(inf, rng.binomial(1, 0.8, n), rng.binomial(1, 0.15, n)),
        "fatigue": np.where(inf, rng.binomial(1, 0.75, n), rng.binomial(1, 0.2, n)),
        "saturation_o2": np.where(inf, rng.normal(93, 3, n), rng.normal(98, 1, n)).clip(80, 100).round(0),
        "leucocytes": np.where(inf, rng.normal(11.5, 3, n), rng.normal(7, 1.5, n)).clip(2, None).round(1),
        "crp": np.where(inf, rng.normal(45, 20, n), rng.normal(4, 3, n)).clip(0, None).round(1),
        TARGET: y,
    })
    df.to_csv(path, index=False)
    return df


class Dataset:
    def __init__(self, path="data/patient_data.csv"):
        df = pd.read_csv(path)
        self.X = df[FEATURES].to_numpy(dtype=float)
        self.y = df[TARGET].to_numpy(dtype=float)

    def split(self, test_size=0.2, seed=0):
        idx = np.random.default_rng(seed).permutation(len(self.y))
        cut = int(len(idx) * (1 - test_size))
        tr, te = idx[:cut], idx[cut:]
        return self.X[tr], self.X[te], self.y[tr], self.y[te]
