# Home Credit Default Risk – MLOps Project

##  Objectif
Développer un modèle de **scoring crédit** qui évalue la probabilité de défaut de remboursement d’un client, à partir de données issues de la compétition [Kaggle - Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data).

Le projet est mené selon une démarche **MLOps complète** : de l’analyse exploratoire à la mise en production du modèle avec suivi des expérimentations, tests, dashboard d’interprétation, et CI/CD via GitHub Actions.

---

## Contexte métier

La société **"Prêt à Dépenser"**, spécialisée dans le crédit à la consommation, cherche à mieux évaluer les risques de non-remboursement chez des clients sans historique bancaire. Elle souhaite :

- Automatiser les décisions d’octroi de crédit.
- Améliorer la **transparence** des décisions pour les conseillers client.
- Fournir un **dashboard interactif** pour visualiser les scores et interpréter les prédictions.

---

## Stack technique

- **Python 3.9+**
- `scikit-learn`, `xgboost`, `pandas`, `numpy`, `seaborn`, `matplotlib`
- `mlflow`, `shap`, `streamlit`
- `cookiecutter-data-science` structure
- `pytest`, `joblib`
- **CI/CD** : GitHub Actions

---

## Organisation du répertoire

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         home_credit and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── home_credit   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes home_credit a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


## Démarche projet

### 1. Analyse exploratoire (`scoring_01_exploration.ipynb`)
- Nettoyage et imputations
- Visualisations : pairplot, heatmap de corrélations, boxplots revenus/âge
- Analyse SHAP globale (importances des variables)

### 2. Modélisation (`scoring_02_modelling.ipynb`)
- Séparation train/test
- Pipelines `scikit-learn` : preprocessing + modèle
- Modèles testés : Logistic Regression, Decision Tree, RandomForest, Gradient Boosting, XGBoost...
- **MLflow** pour tracking des scores `roc_auc`, `f1`, `accuracy`
- Sélection du **meilleur modèle** : XGBoost (ROC AUC = 0.76)

### 3. Dashboard interactif (Streamlit)
- Sélection d’un client
- Affichage du score et probabilité de défaut
- Interprétation locale avec **SHAP waterfall**
- Visualisation des données client

### 4. Entraînement et prédiction (`train.py`, `predict.py`)
- Entraînement du modèle final
- Sauvegarde en `.joblib`
- Fonction d’inférence unitaire

### 5. CI/CD et tests (`.github/`, `tests/`)
- Test de chargement du modèle
- Vérification des transformations
- Intégration automatisée avec GitHub Actions

---

##  Lancer le projet

###  Installation des dépendances

```bash
pip install -r requirements.txt

```
## Support de présentation:
[Home credit default risk presention](https://www.canva.com/design/DAGuqZcOyDU/WQs1O2dCRbaR0TrygC7NQQ/edit?utm_content=DAGuqZcOyDU&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

