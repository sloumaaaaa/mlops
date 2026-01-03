# Guide d'Exécution - Projet MLOps

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.9+
- Git installé
- Compte GitHub (pour CI/CD)

### 1️⃣ Installation de l'Environnement

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

---

## 📊 Étape 1: Créer les 3 Versions de Datasets

### Version 1 - Dataset Original
```bash
python src/data_loader.py --version 1
```
**Résultat**: `data/v1_california_housing.csv` créé (20,640 lignes)

### Version 2 - Dataset Filtré
```bash
python src/data_loader.py --version 2
```
**Résultat**: `data/v2_filtered_housing.csv` créé (10,297 lignes)
- Outliers supprimés (prix > 5.0)
- Focus sur régions côtières

### Version 3 - Feature Engineering
```bash
python src/data_loader.py --version 3
```
**Résultat**: `data/v3_engineered_housing.csv` créé (10,297 lignes, 13 colonnes)
- 4 nouvelles features ajoutées

---

## 🤖 Étape 2: Entraîner les Modèles

### Modèle 1: Baseline Random Forest avec V1
```bash
python src/train.py --data_path data/v1_california_housing.csv --model random_forest --data_version v1
```
**Métriques:**
- RMSE: 0.4059
- R²: 0.9031
- Temps: ~2-3 secondes

### Modèle 2: Gradient Boosting avec V2
```bash
python src/train.py --data_path data/v2_filtered_housing.csv --model gradient_boosting --data_version v2
```
**Métriques:**
- RMSE: 0.4043
- R²: 0.9060

### Modèle 3: Gradient Boosting avec V3
```bash
python src/train.py --data_path data/v3_engineered_housing.csv --model gradient_boosting --data_version v3
```
**Métriques:**
- RMSE: 0.4023 (meilleur modèle)
- R²: 0.9069

---

## 🎯 Étape 3: Optimisation Avancée avec Optuna

### Optimisation Standard (50 trials)
```bash
python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv --model gradient_boosting --n_trials 50 --data_version v3
```

**Temps estimé**: 5-15 minutes
**Résultat**: Meilleurs paramètres sauvegardés dans `results/optuna/`

---

## 📊 Étape 4: Évaluation et Comparaison

### Comparer Tous les Modèles
```bash
python src/evaluate.py --compare_all
```

**Génère**:
- `results/model_comparison.txt` - Rapport texte détaillé
- `results/model_comparison.png` - Graphiques de comparaison

---

## 🌐 Étape 5: Visualiser dans MLflow UI

### Lancer MLflow UI
```bash
python -m mlflow ui --port 5000
```

Puis ouvrir dans votre navigateur: **http://127.0.0.1:5000**

### Dans l'Interface MLflow
- Experiment: "california-housing"
- 5 runs avec toutes les métriques
- Graphiques de comparaison
- Modèles téléchargeables

---

## 🔄 Étape 6: Workflow Automatisé Complet

### Option Automatique (Recommandé)
```bash
python run_complete_workflow.py
```

Cette commande exécute:
1. Création des 3 datasets
2. Entraînement de tous les modèles
3. Évaluation et comparaison
4. Génération de rapports

**Durée totale**: 5-10 minutes

---

## 🤖 Étape 7: GitHub Actions

### Push vers GitHub
```bash
git add .
git commit -m "Update MLOps project"
git push origin dev
```

### Vérifier l'Exécution
Allez sur: https://github.com/sloumaaaaa/mlops/actions

Vous verrez:
- 6 jobs en cours d'exécution
- Tests de qualité de code
- Validation des données
- Entraînement automatique
- Génération de rapports

---

## 📋 Checklist Complète

- [x] Environnement Python créé et activé
- [x] Dépendances installées
- [x] 3 datasets créés
- [x] 3 modèles entraînés
- [x] MLflow tracking actif
- [x] Évaluation complétée
- [x] GitHub Actions configuré
- [x] Documentation à jour

---

## 🎯 Résultats Attendus

**Datasets:**
- V1: 20,640 lignes × 9 colonnes
- V2: 10,297 lignes × 9 colonnes
- V3: 10,297 lignes × 13 colonnes

**Modèles:**
- RandomForest V1: RMSE 0.4059
- GradientBoosting V2: RMSE 0.4043
- GradientBoosting V3: RMSE 0.4023 (meilleur)

**Amélioration**: 0.88% de V1 à V3

---

**Projet MLOps - ESPRIT - Janvier 2026**
