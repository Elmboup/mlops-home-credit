# 📘 Documentation — Exploration & Nettoyage (`scoring_01_exploration.ipynb`)

## Objectif du Notebook

Ce notebook vise à :

* Charger et explorer les données traitées (`train_merged.csv`)
* Réaliser une **analyse exploratoire approfondie (EDA)** sur la cible `TARGET`
* Nettoyer les données et gérer les valeurs manquantes
* Générer des visualisations pertinentes (statistiques et graphiques)
* Préparer le jeu de données pour la modélisation

---

## Fichier utilisé

* `../data/processed/train_merged.csv` (résultat des agrégations multi-tables)

---

## Plan du notebook

| Étape                                  | Description                                                      |
| -------------------------------------- | ---------------------------------------------------------------- |
| 1. Chargement des données              | Lecture de la table fusionnée et aperçu                          |
| 2. Description des variables           | Aperçu statistique global                                        |
| 3. Vérification des valeurs manquantes | Identification des colonnes à fort taux de nullité               |
| 4. Suppression / imputation            | Suppression des colonnes inutilisables, traitement des NA        |
| 5. Analyse de `TARGET`                 | Répartition binaire, déséquilibre de classes                     |
| 6. Analyse univariée                   | Distributions, outliers                                          |
| 7. Analyse bivariée                    | Relations entre `TARGET` et variables clés                       |
| 8. Corrélations                        | Corrélation entre variables numériques                           |
| 9. ydata-profiling                     | Profil interactif automatique                                    |
| 10. Tranches d'âge & revenu            | Visualisation croisée `AMT_INCOME_TOTAL` vs `AGE_BIN` & `TARGET` |
| 11. Pairplot                           | Corrélations visuelles sur 5 features                            |
| 12. Heatmap corrélation                | Top 15 features les plus corrélées à `TARGET`                    |

---

## Méthodes d'analyse utilisées

###  Nettoyage

* **Suppression de colonnes** avec > 50% de valeurs manquantes
* **Imputation médiane** pour les colonnes numériques (dans pipeline)
* **Encodage des catégories manquantes** via `missing` (dans pipeline catégoriel)

###  Exploration visuelle

* **Boxplots** & **barplots** (revenu, crédit, âge)
* **`pairplot()`** de Seaborn pour visualiser des clusters
* **`heatmap()`** pour identifier les relations corrélées aux variables cibles
* **`ydata-profiling`** : profil interactif étendu (valeurs extrêmes, NA, cardinalité)

---

##  Modularisation dans `features.py`

Les visualisations suivantes sont réutilisables via :

```python
from home_credit.features import run_feature_analysis
run_feature_analysis(df)
```

Fonctions documentées :

* `create_age_bins()` : ajoute une colonne `AGE_BIN` pour les tranches d'âge
* `plot_income_by_age_bin()` : boxplot revenu vs TARGET selon âge
* `plot_pairplot()` : scatter + KDE entre plusieurs features
* `plot_correlation_heatmap()` : heatmap des top corrélations

---

##  Output / Résultat

* Un dataset **nettoyé et exploré visuellement**
* Une meilleure compréhension des corrélations avec la cible `TARGET`
* Un **rapport interactif (HTML)** généré avec ydata-profiling
* Des graphiques réutilisables et paramétrables dans le module `features.py`

---

##  Prochaines étapes

1. Lancer `scoring_02_modelling.ipynb` pour la modélisation supervisée
2. Logguer les modèles avec **MLflow**
3. Implémenter **GridSearch**, **SHAP**, et la **prédiction finale**
4. Déployer un **dashboard de scoring (Streamlit ou FastAPI)**

---
