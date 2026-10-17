Virus Diagnosis
Projet simple de classification binaire pour diagnostiquer si un patient est sain ou infecté à partir de caractéristiques cliniques synthétiques.

Objectif
Ce projet montre une pipeline complète d'apprentissage automatique :

génération de données synthétiques,
préparation des données,
entraînement de modèles,
évaluation des performances,
prédiction sur des cas cliniques.
Modèles utilisés
Régression logistique
Réseau de neurones simple
Structure du projet
main.py : point d'entrée principal
core/ : logique des modèles et du dataset
pipeline/ : entraînement et évaluation
app/ : interface de diagnostic simple
utils/ : utilitaires et métriques
data/ : données générées
Prérequis
Python 3.10+
pip
Installation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Lancement
python3 main.py
Exemple de sortie attendue
LogisticRegression {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
  patient 1 -> Infecté
  patient 2 -> Sain
Remarques
Le dataset est généré automatiquement au lancement du programme dans data/patient_data.csv.
