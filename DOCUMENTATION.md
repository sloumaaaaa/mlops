# Projet MLOps Complet - California Housing Price Prediction

## 📋 Vue d'ensemble du projet

Ce projet implémente un pipeline MLOps complet pour la prédiction des prix de l'immobilier en Californie.

### 🎯 Objectifs
- Pipeline ML reproductible et versionné
- Dataset California Housing (données synthétiques)
- Tracking des expériences avec MLflow
- Versioning des données avec DVC
- Automatisation avec GitHub Actions
- Optimisation d'hyperparamètres avec Optuna

---

## 🏗️ Architecture du Projet

```
MLOPS/
│
├── data/                          # Données versionnées
│   ├── v1_california_housing.csv  # Version 1: 20,640 lignes
│   ├── v2_filtered_housing.csv    # Version 2: 10,297 lignes
│   └── v3_engineered_housing.csv  # Version 3: 10,297×13
│
├── src/                           # Code source
│   ├── data_loader.py            # Chargement données
│   ├── preprocessing.py          # Preprocessing
│   ├── train.py                  # Entraînement MLflow
│   ├── hyperparameter_tuning.py  # Optuna
│   └── evaluate.py               # Évaluation
│
├── models/                        # Modèles sauvegardés
├── results/                       # Rapports et graphiques
├── .github/workflows/
│   └── ml_pipeline.yml           # GitHub Actions
│
├── dvc.yaml                       # Pipeline DVC
├── requirements.txt               # Dépendances
├── run_complete_workflow.py      # Workflow automatisé
└── change_dataset.py             # Gestion versions datasets
```

---

## 🔧 Outils Utilisés

### 1. **Git** - Contrôle de Version
- Gestion du code source
- Repository: https://github.com/sloumaaaaa/mlops
- Branch principale: `dev`

### 2. **MLflow** - Tracking des Expériences
- Tracking Server: SQLite backend (mlflow.db)
- Experiment: california-housing
- 5 runs trackés avec métriques complètes

**Fonctionnalités:**
- `mlflow.log_param()`: Hyperparamètres
- `mlflow.log_metric()`: RMSE, MAE, R², MAPE
- `mlflow.log_artifact()`: Graphiques
- `mlflow.sklearn.log_model()`: Modèles

### 3. **DVC (Data Version Control)** - Gestion des Données
- Pipeline configuré (dvc.yaml - 8 stages)
- Versioning des 3 datasets

### 4. **GitHub Actions** - CI/CD
- 6 jobs automatisés
- Triggers: push, PR, schedule, manual
- Matrix strategy: 2 models × 3 versions

### 5. **Optuna** - Optimisation
- Recherche bayésienne
- Intégration MLflow
- Visualisations (Plotly + Kaleido)

---

## 🤖 Modèle de Machine Learning

### Dataset: California Housing Prices
- **Source**: Données synthétiques (fallback pour HTTP 403)
- **Taille**: 20,640 échantillons
- **Features**: 8 caractéristiques
  - MedInc: Revenu médian
  - HouseAge: Âge médian des maisons
  - AveRooms: Nombre moyen de pièces
  - AveBedrms: Nombre moyen de chambres
  - Population: Population du bloc
  - AveOccup: Occupation moyenne
  - Latitude: Latitude
  - Longitude: Longitude
- **Target**: MedHouseVal (prix médian)

### Modèles Testés
1. **Random Forest** (Baseline) - V1
2. **Gradient Boosting** - V2, V3

### Métriques d'Évaluation
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient de détermination
- **MAPE**: Mean Absolute Percentage Error

---

## 📊 Versions des Datasets

### Version 1: Dataset Original
- **Fichier**: `v1_california_housing.csv`
- **Taille**: 20,640 lignes × 9 colonnes
- **Description**: Dataset synthétique complet
- **RMSE**: 0.4059 (RandomForest)

### Version 2: Dataset Filtré
- **Fichier**: `v2_filtered_housing.csv`
- **Taille**: 10,297 lignes × 9 colonnes
- **Transformations**:
  - Outliers supprimés (prix > 5.0)
  - Focus régions côtières
- **RMSE**: 0.4043 (GradientBoosting)
- **Amélioration**: 0.39%

### Version 3: Feature Engineering
- **Fichier**: `v3_engineered_housing.csv`
- **Taille**: 10,297 lignes × 13 colonnes
- **Nouvelles features**:
  - rooms_per_household
  - bedrooms_ratio
  - population_density
  - income_category
- **RMSE**: 0.4023 (GradientBoosting)
- **Amélioration**: 0.88% vs V1

---

## 🚀 Flux de Travail (Workflow)

### 1. Préparation des Données
```bash
python src/data_loader.py --version 1
python src/data_loader.py --version 2
python src/data_loader.py --version 3
```

### 2. Entraînement
```bash
python src/train.py --data_path data/v1_california_housing.csv --model random_forest --data_version v1
python src/train.py --data_path data/v2_filtered_housing.csv --model gradient_boosting --data_version v2
python src/train.py --data_path data/v3_engineered_housing.csv --model gradient_boosting --data_version v3
```

### 3. Optimisation Optuna
```bash
python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv --model gradient_boosting --n_trials 50 --data_version v3
```

### 4. Évaluation
```bash
python src/evaluate.py --compare_all
```

### 5. Visualisation MLflow
```bash
python -m mlflow ui --port 5000
```

### 6. GitHub Actions
Le workflow s'exécute automatiquement sur push vers `dev` ou `main`.

---

## 📈 Résultats et Comparaisons

### Tableau Comparatif

| Run | Model | Data Version | RMSE | R² | MAE | MAPE |
|-----|-------|-------------|------|-----|-----|------|
| 1 | RandomForest | V1 | 0.4059 | 0.9031 | 0.2925 | 11.72% |
| 2 | GradientBoosting | V2 | 0.4043 | 0.9060 | 0.2908 | 11.62% |
| **3** | **GradientBoosting** | **V3** | **0.4023** | **0.9069** | **0.2894** | **11.55%** |

### Insights
- Feature engineering (V3) améliore les performances
- GradientBoosting surpasse RandomForest
- Amélioration totale: 0.88% (V1 → V3)

### Fichiers Générés
- `results/model_comparison.txt`: Rapport détaillé
- `results/model_comparison.png`: Graphiques comparatifs
- `models/*.pkl`: Modèles sauvegardés
- MLflow UI: Interface de tracking

---

## 🔄 Pipeline DVC

Le fichier `dvc.yaml` contient 8 stages:
1. load_data_v1/v2/v3
2. preprocess
3. train
4. evaluate
5. optimize

Exécution:
```bash
dvc repro
dvc metrics show
```

---

## 🤝 GitHub Actions Workflow

### 6 Jobs Configurés

1. **Code Quality**: Black, isort, flake8
2. **Data Validation**: Création et validation des 3 datasets
3. **Model Training**: Matrix strategy (2 models × 3 versions)
4. **Hyperparameter Tuning**: Optuna optimization
5. **Model Evaluation**: Comparaison des modèles
6. **Summary Report**: Génération rapport automatique

### Triggers
- Push sur `main`, `dev`
- Pull requests vers `main`
- Planification: Lundi 2h AM
- Manuel: workflow_dispatch

### Artifacts Générés
- Datasets (retention: 7 jours)
- Modèles (retention: 30 jours)
- Rapports d'évaluation (retention: 30 jours)

---

## 📚 Résumé Technique

### Technologies
- **Python 3.9+**
- **MLflow 2.9+**: Tracking
- **DVC 3.30+**: Data versioning
- **scikit-learn 1.3+**: ML models
- **Optuna 3.4+**: Hyperparameter optimization
- **Plotly 5.17+ & Kaleido 0.2+**: Visualisations
- **GitHub Actions**: CI/CD

### Métriques Clés
- Dataset: 20,640 → 10,297 samples (filtrage)
- Features: 9 → 13 (feature engineering)
- Runs trackés: 5
- Amélioration RMSE: 0.88%
- Temps d'entraînement: <5 secondes/modèle

---

**Projet MLOps - ESPRIT - Janvier 2026**
