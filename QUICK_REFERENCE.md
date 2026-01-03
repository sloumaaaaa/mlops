# 🚀 Quick Reference - Commandes Essentielles

## Installation et Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Créer les Datasets
```bash
python src/data_loader.py --version 1
python src/data_loader.py --version 2
python src/data_loader.py --version 3
```

## Entraîner les Modèles
```bash
# Baseline
python src/train.py --data_path data/v1_california_housing.csv --model random_forest --data_version v1

# Gradient Boosting V2
python src/train.py --data_path data/v2_filtered_housing.csv --model gradient_boosting --data_version v2

# Gradient Boosting V3 (meilleur)
python src/train.py --data_path data/v3_engineered_housing.csv --model gradient_boosting --data_version v3
```

## Optimisation Optuna
```bash
python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv --model gradient_boosting --n_trials 50 --data_version v3
```

## Évaluation
```bash
python src/evaluate.py --compare_all
```

## MLflow UI
```bash
python -m mlflow ui --port 5000
# Ouvrir http://127.0.0.1:5000
```

## Git
```bash
git add .
git commit -m "Message"
git push origin dev
```

## Workflow Complet
```bash
python run_complete_workflow.py
```

---

## 📁 Fichiers Importants

| Fichier | Description |
|---------|-------------|
| `README.md` | Vue d'ensemble |
| `DOCUMENTATION.md` | Documentation technique complète |
| `GUIDE_EXECUTION.md` | Guide pas-à-pas |
| `INSTALLATION.md` | Guide d'installation |
| `dvc.yaml` | Pipeline DVC |
| `requirements.txt` | Dépendances |

---

## 🎯 Checklist Rapide

- [ ] Environnement installé
- [ ] 3 datasets créés
- [ ] 3 modèles entraînés
- [ ] MLflow UI lancé
- [ ] Évaluation complétée
- [ ] GitHub Actions configuré

---

## 📊 Résultats

| Modèle | Dataset | RMSE | R² |
|--------|---------|------|-----|
| RandomForest | V1 | 0.4059 | 0.9031 |
| GradientBoosting | V2 | 0.4043 | 0.9060 |
| **GradientBoosting** | **V3** | **0.4023** | **0.9069** |

**Amélioration**: 0.88% de V1 à V3

---

**Projet MLOps - ESPRIT - Janvier 2026**
