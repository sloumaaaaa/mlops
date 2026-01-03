# California Housing Price Prediction - MLOps Project

[![ML Pipeline CI/CD](https://github.com/sloumaaaaa/mlops/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/sloumaaaaa/mlops/actions)

Un projet MLOps complet démontrant les meilleures pratiques pour le développement, le versioning, et le déploiement de modèles de Machine Learning.

## 🎯 Objectif du Projet

Prédire les prix de l'immobilier en Californie en utilisant un pipeline MLOps professionnel avec:
- ✅ **Git** pour le versioning du code
- ✅ **MLflow** pour le tracking des expériences
- ✅ **DVC** pour le versioning des données
- ✅ **GitHub Actions** pour l'automatisation CI/CD
- ✅ **Optuna** pour l'optimisation d'hyperparamètres

## 📊 Dataset

**California Housing Prices Dataset**
- 20,640 échantillons (données synthétiques)
- 8 features (MedInc, HouseAge, AveRooms, etc.)
- Target: Prix médian des maisons

**3 Versions du Dataset:**
1. **V1**: Dataset original complet (20,640 lignes)
2. **V2**: Dataset filtré (10,297 lignes - outliers supprimés, focus côtier)
3. **V3**: Feature engineering (10,297 lignes × 13 colonnes - 4 nouvelles features)

## 🚀 Installation Rapide

```bash
# Cloner le repository
git clone https://github.com/sloumaaaaa/mlops.git
cd mlops

# Créer environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

## 🏃 Quick Start

### Option 1: Workflow Automatisé Complet
```bash
python run_complete_workflow.py
```

Cette commande exécute automatiquement:
1. Création des 3 versions de datasets
2. Entraînement de plusieurs modèles
3. Évaluation et comparaison des résultats
4. Génération de rapports

### Option 2: Étapes Individuelles

#### 1. Créer les Datasets
```bash
python src/data_loader.py --version 1
python src/data_loader.py --version 2  
python src/data_loader.py --version 3
```

#### 2. Entraîner les Modèles
```bash
# Random Forest avec V1
python src/train.py --data_path data/v1_california_housing.csv --model random_forest --data_version v1

# Gradient Boosting avec V2
python src/train.py --data_path data/v2_filtered_housing.csv --model gradient_boosting --data_version v2

# Gradient Boosting avec V3
python src/train.py --data_path data/v3_engineered_housing.csv --model gradient_boosting --data_version v3
```

#### 3. Optimisation Hyperparamètres (Optuna)
```bash
python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv --model gradient_boosting --n_trials 50 --data_version v3
```

#### 4. Évaluer et Comparer
```bash
python src/evaluate.py --compare_all
```

#### 5. Visualiser dans MLflow UI
```bash
python -m mlflow ui --port 5000
# Ouvrir http://127.0.0.1:5000
```

## 📈 Résultats

| Modèle | Dataset | RMSE | R² | MAE |
|--------|---------|------|-----|-----|
| RandomForest | V1 | 0.4059 | 0.9031 | 0.2925 |
| GradientBoosting | V2 | 0.4043 | 0.9060 | 0.2908 |
| **GradientBoosting** | **V3** | **0.4023** | **0.9069** | **0.2894** |

**Amélioration**: 0.88% de V1 à V3

## 🤖 GitHub Actions

Le workflow CI/CD s'exécute automatiquement sur:
- Push sur branches `main`, `dev`
- Pull requests vers `main`
- Planification hebdomadaire (lundi 2h AM)
- Déclenchement manuel

**6 Jobs automatisés:**
1. Code Quality Check
2. Data Validation
3. Model Training (matrix: 2 models × 3 versions)
4. Hyperparameter Optimization
5. Model Evaluation
6. Summary Report Generation

## 📚 Documentation

- [DOCUMENTATION.md](DOCUMENTATION.md) - Documentation technique complète
- [GUIDE_EXECUTION.md](GUIDE_EXECUTION.md) - Guide d'exécution pas-à-pas
- [INSTALLATION.md](INSTALLATION.md) - Guide d'installation détaillé
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Référence rapide des commandes
- [PROJET_RESUME.md](PROJET_RESUME.md) - Résumé du projet

## 🛠️ Technologies Utilisées

- **Python 3.9+**
- **MLflow** - Experiment tracking
- **DVC** - Data versioning
- **scikit-learn** - ML models
- **Optuna** - Hyperparameter optimization
- **GitHub Actions** - CI/CD
- **Pandas/NumPy** - Data processing
- **Matplotlib/Seaborn/Plotly** - Visualization

## 📁 Structure du Projet

```
MLOPS/
├── src/                        # Code source
│   ├── data_loader.py         # Chargement datasets
│   ├── preprocessing.py       # Preprocessing
│   ├── train.py              # Entraînement
│   ├── hyperparameter_tuning.py  # Optuna
│   └── evaluate.py           # Évaluation
├── data/                      # Datasets (3 versions)
├── models/                    # Modèles sauvegardés
├── results/                   # Rapports et graphiques
├── .github/workflows/         # GitHub Actions
├── dvc.yaml                   # Pipeline DVC
└── requirements.txt           # Dépendances
```

## 🎓 Auteur

Projet MLOps - ESPRIT - Janvier 2026

