"""Point d'entrée du projet virus_diag.

    python main.py          -> teste chaque fonction/classe, puis lance la démo
    python main.py --ui     -> lance l'interface web Flask (http://127.0.0.1:5000)
"""
import os
import sys
import tempfile

import numpy as np

# Les chemins (data/patient_data.csv) sont relatifs au dossier du projet
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from core.dataset import Dataset, generate_patient_data, FEATURES, TARGET
from core.model import Model
from core.optimizer import SGD
from core.logistic_regression import LogisticRegression, sigmoid
from core.neural_network import NeuralNetwork
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from app.interface_clinique import ClinicalPredictor
from app.api import diagnose_patient
from utils import metrics
from utils.preprocessing import StandardScaler

PATIENT_MALADE = {"age": 62, "temperature": 39.1, "toux": 1, "fatigue": 1,
                  "saturation_o2": 92, "leucocytes": 13.0, "crp": 60}
PATIENT_SAIN = {"age": 30, "temperature": 36.8, "toux": 0, "fatigue": 0,
                "saturation_o2": 98, "leucocytes": 6.5, "crp": 2}


# --------------------------------------------------------------------------
# Tests : une fonction par fonction / classe du projet
# --------------------------------------------------------------------------
def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert sigmoid(100) > 0.999 and sigmoid(-100) < 0.001
    assert np.all((sigmoid(np.array([-3, 0, 3])) > 0) & (sigmoid(np.array([-3, 0, 3])) < 1))


def test_sgd():
    p = np.array([1.0, 2.0])
    SGD(lr=0.5).update([p], [np.array([1.0, 1.0])])
    assert np.allclose(p, [0.5, 1.5])          # mise à jour en place


def test_standard_scaler():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    Z = StandardScaler().fit(X).transform(X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-6)


def test_metrics():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 0, 1])
    assert metrics.accuracy(y, pred) == 0.5
    assert metrics.precision(y, pred) == 0.5
    assert metrics.recall(y, pred) == 0.5
    assert abs(metrics.f1(y, pred) - 0.5) < 1e-9
    assert metrics.f1(np.array([0, 0]), np.array([0, 0])) == 0.0


def test_generate_patient_data():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.csv")
        df = generate_patient_data(path=path, n=100, seed=1)
        assert os.path.exists(path) and len(df) == 100
        assert list(df.columns) == FEATURES + [TARGET]
        assert set(df[TARGET].unique()) <= {0, 1}


def test_dataset_split():
    X_tr, X_te, y_tr, y_te = Dataset().split(test_size=0.2)
    assert X_tr.shape[1] == len(FEATURES)
    assert len(X_tr) == len(y_tr) and len(X_te) == len(y_te)
    assert len(X_tr) + len(X_te) == 500 and len(X_te) == 100


def test_model_abstract():
    try:
        Model()
    except TypeError:
        return
    raise AssertionError("Model devrait être abstraite")


def _jeu_simple():
    X_tr, X_te, y_tr, y_te = Dataset().split()
    return X_tr, X_te, y_tr, y_te


def test_logistic_regression():
    X_tr, X_te, y_tr, y_te = _jeu_simple()
    model, scaler = Trainer(LogisticRegression()).train(X_tr, y_tr)
    proba = model.predict(scaler.transform(X_te))
    assert proba.shape == (len(X_te),) and np.all((proba >= 0) & (proba <= 1))
    assert Evaluator(model, scaler).evaluate(X_te, y_te)["accuracy"] > 0.8


def test_neural_network():
    X_tr, X_te, y_tr, y_te = _jeu_simple()
    model, scaler = Trainer(NeuralNetwork()).train(X_tr, y_tr)
    proba = model.predict(scaler.transform(X_te))
    assert proba.shape == (len(X_te),) and np.all((proba >= 0) & (proba <= 1))
    assert Evaluator(model, scaler).evaluate(X_te, y_te)["accuracy"] > 0.8


def test_trainer():
    X_tr, _, y_tr, _ = _jeu_simple()
    model, scaler = Trainer(LogisticRegression(epochs=50)).train(X_tr, y_tr)
    assert isinstance(model, Model) and isinstance(scaler, StandardScaler)
    assert hasattr(model, "w") and hasattr(scaler, "mean")


def test_evaluator():
    X_tr, X_te, y_tr, y_te = _jeu_simple()
    model, scaler = Trainer(LogisticRegression(epochs=100)).train(X_tr, y_tr)
    res = Evaluator(model, scaler).evaluate(X_te, y_te)
    assert set(res) == {"accuracy", "precision", "recall", "f1"}
    assert all(0 <= v <= 1 for v in res.values())


def test_clinical_predictor():
    X_tr, _, y_tr, _ = _jeu_simple()
    model, scaler = Trainer(LogisticRegression()).train(X_tr, y_tr)
    predictor = ClinicalPredictor(model, scaler)
    assert predictor.diagnose([PATIENT_MALADE[f] for f in FEATURES]) == "Infecté"
    assert predictor.diagnose([PATIENT_SAIN[f] for f in FEATURES]) == "Sain"
    assert 0.0 <= predictor.probability([PATIENT_MALADE[f] for f in FEATURES]) <= 1.0


def test_diagnose_patient():
    X_tr, _, y_tr, _ = _jeu_simple()
    model, scaler = Trainer(NeuralNetwork()).train(X_tr, y_tr)
    assert diagnose_patient(model, scaler, PATIENT_MALADE) == "Infecté"
    assert diagnose_patient(model, scaler, PATIENT_SAIN) == "Sain"


def run_tests():
    tests = [v for k, v in sorted(globals().items(), key=lambda kv: list(globals()).index(kv[0]))
             if k.startswith("test_") and callable(v)]
    echecs = 0
    print("=== Tests unitaires ===")
    for t in tests:
        try:
            t()
            print(f"  [OK]    {t.__name__}")
        except Exception as e:                       # noqa: BLE001
            echecs += 1
            print(f"  [ECHEC] {t.__name__} -> {type(e).__name__}: {e}")
    print(f"=> {len(tests) - echecs}/{len(tests)} tests réussis\n")
    return echecs == 0


# --------------------------------------------------------------------------
# Démo (comportement du main d'origine)
# --------------------------------------------------------------------------
def demo():
    print("=== Démo ===")
    generate_patient_data()                      # crée data/patient_data.csv
    X_tr, X_te, y_tr, y_te = Dataset().split()
    # Le même ClinicalPredictor fonctionne avec les deux modèles (polymorphisme)
    for model in (LogisticRegression(), NeuralNetwork()):
        model, scaler = Trainer(model).train(X_tr, y_tr)
        print(type(model).__name__, Evaluator(model, scaler).evaluate(X_te, y_te))
        print("  patient 1 ->", diagnose_patient(model, scaler, PATIENT_MALADE))
        print("  patient 2 ->", diagnose_patient(model, scaler, PATIENT_SAIN))


def main():
    if "--ui" in sys.argv:
        from app.web import create_app
        create_app().run(debug=False, port=5000)
        return
    if not os.path.exists("data/patient_data.csv"):
        generate_patient_data()
    ok = run_tests()
    demo()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
