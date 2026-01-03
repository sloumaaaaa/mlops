# 🚀 Quick Reference - Commandes Essentielles

## Installation et Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
dvc init
```

## Créer les Datasets
```bash
# Tout automatique
python change_dataset.py --all

# Ou un par un
python src/data_loader.py --version 1
python src/data_loader.py --version 2
python src/data_loader.py --version 3
```

## Entraîner les Modèles
```bash
# Baseline
python src/train.py --data_path data/v1_california_housing.csv --model random_forest --data_version v1

# XGBoost V2
python src/train.py --data_path data/v2_filtered_housing.csv --model xgboost --data_version v2

# XGBoost V3
python src/train.py --data_path data/v3_engineered_housing.csv --model xgboost --data_version v3
```

## Optimisation Optuna
```bash
python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv --model xgboost --n_trials 50 --data_version v3
```

## Évaluation
```bash
python src/evaluate.py --compare_all
```

## MLflow UI
```bash
mlflow ui
# Ouvrir http://localhost:5000
```

## DVC
```bash
dvc repro                 # Exécuter pipeline
dvc metrics show          # Voir métriques
dvc status               # Vérifier status
```

## Git
```bash
git add .
git commit -m "Message"
git push origin main
```

## Workflow Complet
```bash
python run_complete_workflow.py
```

## Vérifications
```bash
# Dépendances
python -c "import sklearn, mlflow, dvc, optuna, xgboost; print('✅ OK')"

# Datasets
ls data/*.csv

# Expériences MLflow
mlflow experiments list
```

---

## 📁 Fichiers Importants

| Fichier | Description |
|---------|-------------|
| `PROJET_RESUME.md` | Résumé complet du projet |
| `DOCUMENTATION.md` | Documentation détaillée (90+ pages) |
| `GUIDE_EXECUTION.md` | Guide pas-à-pas |
| `README.md` | Vue d'ensemble |
| `dvc.yaml` | Pipeline DVC |
| `requirements.txt` | Dépendances |

---

## 🎯 Checklist Rapide

- [ ] Environnement installé
- [ ] DVC initialisé
- [ ] 3 datasets créés
- [ ] Au moins 3 modèles entraînés
- [ ] Optuna exécuté
- [ ] MLflow UI visualisé
- [ ] Rapport de comparaison généré

---

## 💡 Résultats Attendus

| Version | RMSE | Amélioration |
|---------|------|--------------|
| V1 | ~0.52 | Baseline |
| V2 | ~0.45 | -15% |
| V3 | ~0.41 | -22% |
| V3+Optuna | ~0.39 | -25% |

---

**Besoin d'aide?** Consultez `GUIDE_EXECUTION.md`
