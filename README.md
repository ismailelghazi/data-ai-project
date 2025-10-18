# Prédiction du temps de livraison / Delivery Time Prediction
<div align="center"> <img src="https://www.simplon.ma/images/Simplon_Maghreb_Rouge.png" alt="Simplon Maghreb Logo" width="250" style="transform: rotateY(15deg) rotateX(5deg); box-shadow: 0 4px 20px rgba(0,0,0,0.3); border-radius: 10px;"> <div> <img src="https://img.shields.io/badge/-Python-black?style=for-the-badge&logo=python&logoColor=white&color=3776AB" /> <img src="https://img.shields.io/badge/-Pandas-black?style=for-the-badge&logo=pandas&logoColor=white&color=150458" /> <img src="https://img.shields.io/badge/-NumPy-black?style=for-the-badge&logo=numpy&logoColor=white&color=013243" /> <img src="https://img.shields.io/badge/-Scikit--Learn-black?style=for-the-badge&logo=scikitlearn&logoColor=white&color=F7931E" /> <img src="https://img.shields.io/badge/-Matplotlib-black?style=for-the-badge&logo=plotly&logoColor=white&color=11557C" /> <img src="https://img.shields.io/badge/-Seaborn-black?style=for-the-badge&logoColor=white&color=4C9F70" /> <img src="https://img.shields.io/badge/-Jupyter-black?style=for-the-badge&logo=jupyter&logoColor=white&color=F37626" /> <img src="https://img.shields.io/badge/-Git-black?style=for-the-badge&logo=git&logoColor=white&color=F05032" /> <img src="https://img.shields.io/badge/-Jira-black?style=for-the-badge&logo=jira&logoColor=white&color=0052CC" /> </div> </div>
## Résumé / Project Summary
Ce projet est une preuve de concept (PoC) pour une entreprise de logistique. L'objectif est de prédire le temps total de livraison (DeliveryTime) pour :
- Anticiper les retards
- Informer les clients en temps réel
- Optimiser l’organisation des tournées

This project is a proof of concept (PoC) for a logistics company. The goal is to predict total delivery time to anticipate delays, inform customers in real time, and optimize delivery routes.

## Contexte / Context
Les estimations de livraison sont actuellement manuelles et les retards fréquents entraînent une insatisfaction client. L’équipe souhaite une solution automatisée, fiable et prête à être validée avant déploiement.

Delivery time estimates are currently manual and frequent delays lead to customer dissatisfaction. The team wants a reliable, automated solution before deployment.

## Objectifs / Objectives
- Construire un modèle de régression pour prédire DeliveryTime
- Automatiser le prétraitement, la sélection de features et la modélisation via un pipeline sklearn
- Comparer RandomForestRegressor et SVR avec GridSearchCV (métrique: MAE)
- Fournir des tests automatisés et configurer une CI (GitHub Actions)

## Variables prises en compte / Features
- Distance_km (numérique) — distance entre restaurant et adresse  
- Traffic_Level (catégoriel/ordinal) — niveau de trafic  
- Vehicle_Type (catégoriel) — type de véhicule  
- Time_of_Day (catégoriel) — plage horaire  
- Courier_Experience (numérique/catégoriel) — expérience du livreur  
- Weather (catégoriel) — conditions météo  
- Preparation_time (numérique) — temps de préparation  
- DeliveryTime (numérique) — cible (target)

## Pipeline proposé / Proposed pipeline
1. EDA : corrélations (heatmap), distributions, countplots, boxplots (ex. trafic vs DeliveryTime)  
2. Prétraitement : imputation si nécessaire, StandardScaler (numériques), OneHotEncoder (catégorielles)  
3. Sélection de features : SelectKBest (f_regression)  
4. Modélisation : GridSearchCV (cv=5) sur RandomForestRegressor et SVR, scoring = MAE  
5. Pipeline sklearn : prétraitement → SelectKBest → meilleur modèle  
6. Tests automatisés (pytest) + CI (GitHub Actions)

## Résultats — Choix du modèle / Model selection
Performance sur le jeu de test :
- Random Forest — R²: 0.844, MAE: 6.52  
- SVR — R²: 0.568, MAE: 10.944

Conclusion : Random Forest est retenu (MAE plus faible, R² plus élevé). Avantages : robuste, capture les non-linéarités, moins sensible à l’échelle, interprétable via feature importance.

## Recommandations opérationnelles / Operational recommendations
- Déployer le pipeline complet : prétraitement → SelectKBest → RandomForest (sérialiser avec joblib)  
- Mettre en place une surveillance : MAE en production et détection de dérive des features — alerter si MAE > 10 minutes  
- Expliquer les prédictions (SHAP) pour la transparence métier  
- Versionner les modèles & hyperparamètres et planifier des ré-entraînements (ou sur drift)

## Commandes utiles / Useful commands
Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Entraînement
```bash
python src/train.py --data data/delivery_data.csv --output models/best_pipeline.joblib
```

Prédiction
```bash
python src/predict.py --model models/best_pipeline.joblib --input samples/sample.json
```

Tests
```bash
pytest -v
```

## CI (exemple rapide)
Ajouter un workflow GitHub Actions pour lancer les tests à chaque push sur main (voir .github/workflows/python-tests.yml).

## Contact
Project owner: ismailelghazi
