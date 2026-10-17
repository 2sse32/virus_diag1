"""Interface web Flask : saisie d'un patient -> diagnostic + probabilité."""
import os

from flask import Flask, render_template, request

from app.interface_clinique import ClinicalPredictor
from core.dataset import Dataset, generate_patient_data, FEATURES
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from pipeline.evaluator import Evaluator
from pipeline.trainer import Trainer

# nom affiché -> (classe, valeurs par défaut du formulaire)
MODELES = {"Régression logistique": LogisticRegression, "Réseau de neurones": NeuralNetwork}

CHAMPS = [  # (nom, libellé, type, min, max, step, défaut)
    ("age", "Âge (ans)", "number", 0, 120, 1, 45),
    ("temperature", "Température (°C)", "number", 34, 42, 0.1, 37.0),
    ("saturation_o2", "Saturation O₂ (%)", "number", 70, 100, 1, 97),
    ("leucocytes", "Leucocytes (10³/µL)", "number", 0, 50, 0.1, 7.0),
    ("crp", "CRP (mg/L)", "number", 0, 500, 0.1, 5.0),
]
BOOLEENS = [("toux", "Toux"), ("fatigue", "Fatigue")]


def entrainer_modeles():
    """Entraîne les deux modèles une seule fois au démarrage."""
    if not os.path.exists("data/patient_data.csv"):
        generate_patient_data()
    X_tr, X_te, y_tr, y_te = Dataset().split()
    resultat = {}
    for nom, classe in MODELES.items():
        model, scaler = Trainer(classe()).train(X_tr, y_tr)
        resultat[nom] = {"predictor": ClinicalPredictor(model, scaler),
                         "metrics": Evaluator(model, scaler).evaluate(X_te, y_te)}
    return resultat


def create_app():
    app = Flask(__name__)
    modeles = entrainer_modeles()

    @app.route("/", methods=["GET", "POST"])
    def index():
        valeurs = {c[0]: c[6] for c in CHAMPS}
        valeurs.update({k: 0 for k, _ in BOOLEENS})
        choix = next(iter(modeles))
        resultat, erreur = None, None

        if request.method == "POST":
            choix = request.form.get("modele", choix)
            try:
                for k in valeurs:
                    if k in dict(BOOLEENS):
                        valeurs[k] = 1 if request.form.get(k) else 0
                    else:
                        valeurs[k] = float(request.form[k])
                if choix not in modeles:
                    raise ValueError("Modèle inconnu.")
                predictor = modeles[choix]["predictor"]
                x = [valeurs[f] for f in FEATURES]
                proba = predictor.probability(x)
                resultat = {"diagnostic": predictor.diagnose(x), "proba": round(proba * 100, 1)}
            except (KeyError, ValueError) as e:
                erreur = f"Valeurs invalides : {e}"

        return render_template("index.html", champs=CHAMPS, booleens=BOOLEENS,
                               valeurs=valeurs, modeles=modeles, choix=choix,
                               resultat=resultat, erreur=erreur)

    return app
