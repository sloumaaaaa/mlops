# Guide d'Exécution - Projet MLOps

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.9+
- Git installé
- (Optionnel) Compte GitHub pour le CI/CD

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

### 2️⃣ Initialisation du Projet

```bash
# Initialiser Git (si pas déjà fait)
git init
git add .
git commit -m "Initial commit - MLOps project setup"

# Initialiser DVC
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
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
**Résultat**: `data/v2_filtered_housing.csv` créé (~18,500 lignes)
- Outliers supprimés (prix > 5.0)
- Focus sur régions côtières

### Version 3 - Feature Engineering
```bash
python src/data_loader.py --version 3
```
**Résultat**: `data/v3_engineered_housing.csv` créé (~18,500 lignes, 13 colonnes)
- 4 nouvelles features ajoutées

---

## 🔄 Étape 2: Versionner les Datasets avec DVC

### Méthode Automatique (Recommandée)
```bash
python change_dataset.py --all
```
Cette commande:
1. Crée les 3 versions
2. Track avec DVC (`dvc add`)
3. Commit dans Git avec messages descriptifs

### Méthode Manuelle
```bash
# Pour chaque dataset
dvc add data/v1_california_housing.csv
git add data/v1_california_housing.csv.dvc .gitignore
git commit -m "Add dataset V1 - Original data"

dvc add data/v2_filtered_housing.csv
git add data/v2_filtered_housing.csv.dvc .gitignore
git commit -m "Add dataset V2 - Filtered data"

dvc add data/v3_engineered_housing.csv
git add data/v3_engineered_housing.csv.dvc .gitignore
git commit -m "Add dataset V3 - Feature engineered"
```

### Vérifier le Versioning
```bash
# Voir les fichiers DVC
ls data/*.dvc

# Voir l'historique Git
git log --oneline

# Statut DVC
dvc status
```

---

## 🤖 Étape 3: Entraîner les Modèles

### Modèle 1: Baseline Random Forest avec V1
```bash
python src/train.py --data_path data/v1_california_housing.csv --model random_forest --data_version v1
```
**Métriques attendues**:
- RMSE: ~0.52
- R²: ~0.80
- Temps: ~2-3 secondes

### Modèle 2: XGBoost avec V1
```bash
python src/train.py --data_path data/v1_california_housing.csv --model xgboost --data_version v1
```
**Métriques attendues**:
- RMSE: ~0.49
- R²: ~0.82

### Modèle 3: XGBoost avec V2 (Données Filtrées)
```bash
python src/train.py --data_path data/v2_filtered_housing.csv --model xgboost --data_version v2
```
**Métriques attendues**:
- RMSE: ~0.45 ⬇️ (-15% vs V1)
- R²: ~0.85

### Modèle 4: XGBoost avec V3 (Feature Engineering)
```bash
python src/train.py --data_path data/v3_engineered_housing.csv --model xgboost --data_version v3
```
**Métriques attendues**:
- RMSE: ~0.41 ⬇️ (-22% vs V1)
- R²: ~0.88

### Avec Hyperparamètres Personnalisés
```bash
python src/train.py --data_path data/v3_engineered_housing.csv --model xgboost --data_version v3 --n_estimators 200 --max_depth 8 --learning_rate 0.05
```

---

## 🎯 Étape 4: Optimisation Avancée avec Optuna

### Optimisation Standard (50 trials)
```bash
python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv --model xgboost --n_trials 50 --data_version v3
```

### Optimisation Intensive (100 trials)
```bash
python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv --model xgboost --n_trials 100 --data_version v3
```

**Temps estimé**: 5-15 minutes selon le CPU
**Résultat**: 
- Fichier `results/optuna/xgboost_best_params.json` avec les meilleurs paramètres
- Amélioration de 3-5% du RMSE vs paramètres par défaut

### Voir les Résultats Optuna
```bash
cat results/optuna/xgboost_best_params.json
```

---

## 📊 Étape 5: Évaluation et Comparaison

### Comparer Tous les Modèles
```bash
python src/evaluate.py --compare_all
```

**Génère**:
- `results/comparison_by_version.png` - Comparaison par version de dataset
- `results/comparison_by_model.png` - Comparaison par type de modèle
- `results/comparison_report.txt` - Rapport textuel détaillé

### Voir le Top 5 des Modèles
```bash
python src/evaluate.py
```

---

## 🌐 Étape 6: Visualiser dans MLflow UI

### Lancer MLflow UI
```bash
mlflow ui
```

Puis ouvrir dans votre navigateur: **http://localhost:5000**

### Dans l'Interface MLflow

**Expériences disponibles**:
- `california-housing` - Tous les entraînements normaux
- `optuna-tuning` - Résultats d'optimisation Optuna

**Fonctionnalités**:
1. **Tableau des runs** - Voir tous les runs avec métriques
2. **Comparaison** - Sélectionner plusieurs runs et cliquer "Compare"
3. **Graphiques** - Voir l'évolution des métriques
4. **Artifacts** - Télécharger modèles et graphiques
5. **Paramètres** - Voir tous les hyperparamètres utilisés

### Comparer des Runs
1. Cocher 2-3 runs dans la liste
2. Cliquer sur "Compare"
3. Analyser les différences de paramètres et métriques

---

## 🔄 Workflow DVC Complet

### Exécuter le Pipeline DVC
```bash
# Exécuter toutes les étapes
dvc repro

# Exécuter une étape spécifique
dvc repro train_v3
```

### Voir les Métriques DVC
```bash
dvc metrics show
```

### Comparer des Versions
```bash
# Comparer avec le commit précédent
dvc metrics diff HEAD~1
```

---

## ⚙️ Workflow Automatisé Complet

### Tout en Une Commande
```bash
python run_complete_workflow.py
```

**Cette commande exécute automatiquement**:
1. ✅ Création des 3 datasets
2. ✅ Entraînement de 4 modèles différents
3. ✅ Optimisation Optuna (50 trials)
4. ✅ Évaluation et comparaison complète

**Temps total**: 15-30 minutes

---

## 📈 Analyser les Résultats

### 1. Rapport de Comparaison
```bash
cat results/comparison_report.txt
```

### 2. Graphiques
Les graphiques sont sauvegardés dans:
- `artifacts/predictions_*.png` - Prédictions vs valeurs réelles
- `results/comparison_by_version.png` - Boxplots par version
- `results/comparison_by_model.png` - Boxplots par modèle

### 3. Meilleurs Paramètres Optuna
```bash
cat results/optuna/xgboost_best_params.json
```

Exemple de sortie:
```json
{
  "n_estimators": 245,
  "max_depth": 7,
  "learning_rate": 0.089,
  "min_child_weight": 3,
  "subsample": 0.87,
  "colsample_bytree": 0.92,
  "gamma": 1.2,
  "reg_alpha": 0.15,
  "reg_lambda": 0.34
}
```

---

## 🚀 GitHub Actions (CI/CD)

### Configuration

1. Créer un repository GitHub
2. Pusher votre code:
```bash
git remote add origin https://github.com/votre-username/mlops.git
git branch -M main
git push -u origin main
```

3. GitHub Actions se déclenchera automatiquement!

### Pipeline Automatique

Le workflow `.github/workflows/ml_pipeline.yml` exécute:

**Job 1: Code Quality**
- Black formatting check
- Flake8 linting
- Import sorting (isort)

**Job 2: Data Validation**
- Chargement des 3 versions de datasets
- Validation de la structure des données

**Job 3: Model Training** (Matrix)
- Entraîne 6 combinaisons (2 modèles × 3 versions)
- Parallélisation automatique

**Job 4: Hyperparameter Tuning**
- Optuna avec 20 trials (limité pour CI)

**Job 5: Model Evaluation**
- Génération des rapports de comparaison

**Job 6: Summary**
- Résumé publié dans GitHub Actions UI

### Voir les Résultats

1. Aller sur GitHub → Actions
2. Cliquer sur le dernier workflow run
3. Voir les jobs et leurs résultats
4. Télécharger les artifacts (modèles, graphiques)

---

## 🔍 Vérifications et Diagnostics

### Vérifier l'Installation
```bash
python -c "import sklearn, mlflow, dvc, optuna, xgboost; print('✅ All dependencies OK')"
```

### Vérifier DVC
```bash
dvc version
dvc status
```

### Vérifier Git
```bash
git status
git log --oneline -5
```

### Vérifier les Datasets
```bash
python -c "import pandas as pd; print('V1:', pd.read_csv('data/v1_california_housing.csv').shape); print('V2:', pd.read_csv('data/v2_filtered_housing.csv').shape); print('V3:', pd.read_csv('data/v3_engineered_housing.csv').shape)"
```

### Lister les Expériences MLflow
```bash
python -c "import mlflow; client = mlflow.tracking.MlflowClient(); exps = client.search_experiments(); print('Experiments:', [e.name for e in exps])"
```

---

## 📋 Checklist du Projet

Utilisez cette checklist pour vérifier que tout est complété:

- [ ] ✅ **Installation**: Environnement virtuel + dépendances installées
- [ ] ✅ **Git**: Repository initialisé et commit initial
- [ ] ✅ **DVC**: Initialisé avec `dvc init`
- [ ] ✅ **Dataset V1**: Créé et versionné avec DVC
- [ ] ✅ **Dataset V2**: Créé et versionné avec DVC
- [ ] ✅ **Dataset V3**: Créé et versionné avec DVC (total 3+ versions ✓)
- [ ] ✅ **Entraînement V1**: Au moins 1 modèle entraîné avec V1
- [ ] ✅ **Entraînement V2**: Au moins 1 modèle entraîné avec V2
- [ ] ✅ **Entraînement V3**: Au moins 1 modèle entraîné avec V3
- [ ] ✅ **MLflow UI**: Lancé et visualisé les expériences
- [ ] ✅ **Optuna**: Optimisation exécutée (fonctionnalité avancée ✓)
- [ ] ✅ **Comparaison**: Rapport généré avec `evaluate.py`
- [ ] ✅ **GitHub Actions**: Workflow configuré (optionnel)
- [ ] ✅ **Documentation**: DOCUMENTATION.md lu et compris

---

## 🎯 Objectifs du Projet - Validation

### Exigences du Projet
1. ✅ **Git**: Code versionné ✓
2. ✅ **MLflow**: Expériences trackées ✓
3. ✅ **DVC**: Données versionnées ✓
4. ✅ **Automation**: GitHub Actions configuré ✓
5. ✅ **Dataset réel**: California Housing (20k+ samples) ✓
6. ✅ **Document descriptif**: DOCUMENTATION.md ✓
7. ✅ **Fonctionnalité avancée**: Optuna pour optimisation ✓
8. ✅ **Changement de dataset 2+ fois**: 3 versions (V1, V2, V3) ✓
9. ✅ **Résultats comparatifs**: Rapports et graphiques ✓

---

## 💡 Conseils et Astuces

### Optimiser le Temps d'Exécution
- Réduire `n_trials` pour Optuna (20-50 au lieu de 100)
- Utiliser des modèles plus simples pour les tests
- Paralléliser avec `n_jobs=-1` (déjà configuré)

### Debugger
```bash
# Vérifier les logs détaillés
python src/train.py --data_path data/v1_california_housing.csv --model xgboost --data_version v1 2>&1 | tee training.log

# Mode verbose pour DVC
dvc repro -v
```

### Sauvegarder le Travail
```bash
# Commit réguliers
git add .
git commit -m "Update: description des changements"

# Push vers GitHub (si configuré)
git push origin main
```

---

## 📞 Support

Si vous rencontrez des problèmes:

1. Vérifiez les logs d'erreur
2. Consultez DOCUMENTATION.md
3. Vérifiez que toutes les dépendances sont installées
4. Assurez-vous que l'environnement virtuel est activé

---

**Projet créé pour le cours MLOps - ESPRIT - Janvier 2026**
