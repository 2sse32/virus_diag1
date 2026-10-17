# 🦠 Virus Diag

Projet Python d'aide au diagnostic. À partir de 7 variables cliniques, il prédit si un patient est **infecté** ou **sain**.

Les deux modèles (régression logistique et réseau de neurones) sont codés **à la main avec NumPy**, sans scikit-learn.

> ⚠️ Projet pédagogique entraîné sur des données **synthétiques**. Il ne remplace en aucun cas un avis médical.

---

## Fonctionnalités

- Deux modèles interchangeables : régression logistique et réseau de neurones (1 couche cachée)
- Entraînement par descente de gradient (`SGD`)
- Évaluation : accuracy, précision, rappel et F1
- Tests unitaires pour chaque fonction, lancés avec une seule commande
- Interface web Flask pour diagnostiquer un patient

---

## Installation

```bash
git clone https://github.com/VOTRE_PSEUDO/virus_diag.git
cd virus_diag
pip install -r requirements.txt
```

Python 3.9 ou plus est recommandé.

---

## Utilisation

### Lancer les tests et la démo

```bash
python main.py
```

Cette commande :
1. génère le jeu de données `data/patient_data.csv` si besoin ;
2. exécute les tests de chaque fonction (`[OK]` / `[ECHEC]`) ;
3. entraîne les deux modèles, affiche leurs métriques et deux exemples de diagnostic.

### Lancer l'interface web

```bash
python main.py --ui
```

Ouvrez ensuite http://127.0.0.1:5000 dans votre navigateur. Vous pouvez :
- saisir les données du patient ;
- choisir le modèle ;
- obtenir le diagnostic et la probabilité d'infection ;
- comparer les performances des deux modèles.

---

## Données d'entrée

| Variable | Description | Type |
|---|---|---|
| `age` | Âge en années | nombre |
| `temperature` | Température corporelle (°C) | nombre |
| `toux` | Présence de toux | 0 ou 1 |
| `fatigue` | Présence de fatigue | 0 ou 1 |
| `saturation_o2` | Saturation en oxygène (%) | nombre |
| `leucocytes` | Globules blancs (10³/µL) | nombre |
| `crp` | Protéine C-réactive (mg/L) | nombre |

**Sortie** : `Infecté` si la probabilité prédite est supérieure ou égale à 0,5, sinon `Sain`.

---

## Structure du projet

```
virus_diag/
├── main.py                      # Tests de chaque fonction + démo + lancement de l'UI
├── requirements.txt
├── data/
│   └── patient_data.csv         # Données synthétiques (500 patients)
├── core/                        # Cœur du machine learning
│   ├── dataset.py               # Génération, chargement et découpage des données
│   ├── model.py                 # Classe abstraite Model (fit / predict)
│   ├── logistic_regression.py   # Régression logistique
│   ├── neural_network.py        # Réseau de neurones
│   └── optimizer.py             # Descente de gradient (SGD)
├── pipeline/
│   ├── trainer.py               # Entraînement (normalisation + fit)
│   └── evaluator.py             # Calcul des métriques
├── utils/
│   ├── preprocessing.py         # StandardScaler
│   └── metrics.py               # accuracy, precision, recall, f1
└── app/
    ├── interface_clinique.py    # ClinicalPredictor : diagnostic d'un patient
    ├── api.py                   # diagnose_patient(model, scaler, patient)
    ├── web.py                   # Application Flask
    └── templates/index.html     # Page web
```

---

## Utiliser le code dans vos scripts

```python
from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from pipeline.trainer import Trainer
from app.api import diagnose_patient

X_tr, X_te, y_tr, y_te = Dataset().split()
model, scaler = Trainer(LogisticRegression()).train(X_tr, y_tr)

patient = {"age": 62, "temperature": 39.1, "toux": 1, "fatigue": 1,
           "saturation_o2": 92, "leucocytes": 13.0, "crp": 60}
print(diagnose_patient(model, scaler, patient))   # -> Infecté
```

Pour utiliser le réseau de neurones, remplacez `LogisticRegression()` par `NeuralNetwork()`. Le reste du code ne change pas, car les deux modèles héritent de `Model`.

---

## Remarque sur les résultats

Les données sont générées avec des profils très différents entre patients sains et infectés. Les modèles obtiennent donc des scores très élevés (proches de 100 %). Ces résultats ne reflètent pas ce que l'on obtiendrait sur des données médicales réelles.

---

## Dépendances

- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [Flask](https://flask.palletsprojects.com/)
