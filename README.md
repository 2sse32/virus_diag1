# 🦠 Virus Diag

Projet Python d'aide au diagnostic : à partir de 7 variables cliniques (âge, température, toux, fatigue, saturation O₂, leucocytes, CRP), il prédit si un patient est **infecté** ou **sain**.

- Modèles implémentés sans bibliothèque de ML : régression logistique et réseau de neurones à une couche cachée, entraînés par descente de gradient
- Architecture orientée objet : classe abstraite `Model`, `Trainer`, `Evaluator` et `ClinicalPredictor` (polymorphisme)
- Tests unitaires pour chaque fonction, lancés avec `python main.py`
- Interface web Flask : `python main.py --ui`

> Projet pédagogique sur données synthétiques, qui ne remplace pas un avis médical.
