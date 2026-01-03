### Option 2: Étape par Étape

#### 1. Créer les Datasets
```bash
# Version 1 - Original
python src/data_loader.py --version 1

# Version 2 - Filtré
python src/data_loader.py --version 2

# Version 3 - Feature Engineering
python src/data_loader.py --version 3
```

#### 2. Versionner avec DVC
```bash
# Automatique avec le script
python change_dataset.py --all

# Ou manuellement pour chaque version
dvc add data/v1_california_housing.csv
git add data/v1_california_housing.csv.dvc .gitignore
git commit -m "Add dataset V1"
```

#### 3. Entraîner les Modèles
```bash
# Baseline avec V1
python src/train.py --data_path data/v1_california_housing.csv --model random_forest --data_version v1

# XGBoost avec V2
python src/train.py --data_path data/v2_filtered_housing.csv --model xgboost --data_version v2

# XGBoost optimisé avec V3
python src/train.py --data_path data/v3_engineered_housing.csv --model xgboost --data_version v3
```

#### 4. Optimisation Avancée (Optuna)
```bash
python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv --model xgboost --n_trials 100 --data_version v3
```

#### 5. Évaluation et Comparaison
```bash
python src/evaluate.py --compare_all
```

#### 6. Visualiser dans MLflow
```bash
mlflow ui
# Ouvrir http://localhost:5000
```

## 📁 Structure du Projet

```
MLOPS/
├── src/                          # Code source
│   ├── data_loader.py           # Chargement des données
│   ├── preprocessing.py         # Préprocessing
│   ├── train.py                 # Entraînement avec MLflow
│   ├── hyperparameter_tuning.py # Optuna optimization
│   └── evaluate.py              # Évaluation et comparaison
├── data/                         # Datasets versionnés avec DVC
├── models/                       # Modèles entraînés
├── .github/workflows/            # GitHub Actions CI/CD
│   └── ml_pipeline.yml
├── dvc.yaml                      # Pipeline DVC
├── requirements.txt              # Dépendances Python
├── DOCUMENTATION.md              # Documentation complète
├── run_complete_workflow.py      # Script automatisé
└── change_dataset.py             # Gestion des versions de datasets
```

## 📈 Résultats Attendus

### Comparaison des Versions de Datasets

| Version | RMSE | R² | Amélioration vs V1 |
|---------|------|----|--------------------|
| V1 (Original) | ~0.52 | ~0.80 | Baseline |
| V2 (Filtré) | ~0.45 | ~0.85 | -15% RMSE |
| V3 (Feature Eng.) | ~0.41 | ~0.88 | -22% RMSE |
| V3 + Optuna | ~0.39 | ~0.89 | -25% RMSE ✨ |

## 🎯 Fonctionnalité Avancée: Optuna

L'optimisation bayésienne avec Optuna apporte:
- 🔍 Recherche intelligente d'hyperparamètres
- ⚡ Pruning automatique des essais non prometteurs
- 📊 Visualisations interactives
- 🔗 Intégration directe avec MLflow
- 📈 Amélioration de 3-5% des performances

## 📚 Documentation Complète

Consultez [DOCUMENTATION.md](DOCUMENTATION.md) pour les détails complets sur:
- Architecture du projet
- Flux de travail détaillé
- Explications des outils
- Résultats et comparaisons
- Guide de troubleshooting

## 🎓 Concepts MLOps Démontrés

1. ✅ **Versioning du Code** (Git)
2. ✅ **Versioning des Données** (DVC) - 3+ versions
3. ✅ **Tracking des Expériences** (MLflow)
4. ✅ **Automatisation CI/CD** (GitHub Actions)
5. ✅ **Optimisation Avancée** (Optuna)
6. ✅ **Reproductibilité** (DVC pipelines)
7. ✅ **Comparaison de Modèles** (Évaluation systématique)

---

⭐ **Consultez DOCUMENTATION.md pour le guide complet!**
