# 🚀 Installation et Premiers Pas - Windows

## ⚡ Installation Rapide (5 minutes)

### 1️⃣ Vérifier Python
```powershell
python --version
# Doit afficher Python 3.9 ou supérieur
```

### 2️⃣ Créer l'Environnement Virtuel
```powershell
# Naviguer vers le dossier du projet
cd C:\Users\sziedi\Desktop\esprit\mlops\MLOPS

# Créer l'environnement
python -m venv venv

# Activer l'environnement
.\venv\Scripts\Activate.ps1
```

**Note**: Si vous avez une erreur "execution policy", exécutez:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3️⃣ Installer les Dépendances
```powershell
# Mettre à jour pip
python -m pip install --upgrade pip

# Installer toutes les bibliothèques
pip install -r requirements.txt
```

**Temps estimé**: 3-5 minutes

### 4️⃣ Vérifier l'Installation
```powershell
python -c "import sklearn, mlflow, dvc, optuna, xgboost, pandas, numpy; print('✅ Toutes les dépendances sont installées!')"
```

---

## 🎯 Premier Test (2 minutes)

### Créer le Premier Dataset
```powershell
python src/data_loader.py --version 1
```

**Résultat attendu**:
```
2026-01-03 10:30:15 - INFO - Loading California Housing dataset...
2026-01-03 10:30:15 - INFO - Dataset loaded: 20640 rows, 9 columns
2026-01-03 10:30:15 - INFO - Creating Version 1: Original dataset
2026-01-03 10:30:16 - INFO - Version 1 saved to data/v1_california_housing.csv
✅ Success!
```

### Entraîner le Premier Modèle
```powershell
python src/train.py --data_path data/v1_california_housing.csv --model random_forest --data_version v1
```

**Résultat attendu**:
```
Training Random Forest...
Training completed in 2.34 seconds
Evaluation Metrics:
  RMSE: 0.5240
  MAE: 0.3650
  R²: 0.8020
  MAPE: 17.45%
✅ Model saved!
```

### Visualiser dans MLflow
```powershell
mlflow ui
```

Puis ouvrez votre navigateur: **http://localhost:5000**

---

## 📋 Workflow Complet Automatisé

### Option la Plus Simple
```powershell
# Tout exécuter automatiquement (15-30 minutes)
python run_complete_workflow.py
```

Ce script va:
1. ✅ Créer les 3 versions de datasets
2. ✅ Entraîner 4 modèles différents
3. ✅ Optimiser avec Optuna (50 trials)
4. ✅ Générer les rapports de comparaison

**Puis visualisez les résultats**:
```powershell
mlflow ui
```

---

## 🔧 Commandes DVC

### Initialiser DVC (première fois uniquement)
```powershell
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### Versionner les Datasets
```powershell
# Option automatique
python change_dataset.py --all

# Ou manuellement
dvc add data/v1_california_housing.csv
git add data/v1_california_housing.csv.dvc .gitignore
git commit -m "Add dataset V1"
```

---

## 📊 Voir les Résultats

### 1. Rapports de Comparaison
```powershell
# Générer les rapports
python src/evaluate.py --compare_all

# Voir le rapport textuel
cat results/comparison_report.txt

# Les graphiques sont dans results/*.png
```

### 2. MLflow UI
```powershell
mlflow ui
```
- Comparer les runs
- Voir les métriques
- Télécharger les modèles

### 3. Meilleurs Paramètres Optuna
```powershell
cat results/optuna/xgboost_best_params.json
```

---

## 🐛 Résolution de Problèmes

### Erreur: "Module not found"
```powershell
# Vérifier que l'environnement est activé
# Vous devez voir (venv) au début de votre ligne de commande

# Si non activé:
.\venv\Scripts\Activate.ps1

# Réinstaller les dépendances
pip install -r requirements.txt
```

### Erreur: "MLflow UI ne démarre pas"
```powershell
# Essayer un autre port
mlflow ui --port 5001
```

### Erreur: "DVC command not found"
```powershell
# DVC doit être installé dans l'environnement virtuel
pip install dvc
```

### Dataset non créé
```powershell
# Vérifier que le dossier data/ existe
mkdir data -Force

# Recréer le dataset
python src/data_loader.py --version 1
```

---

## 📁 Structure Attendue Après Installation

```
MLOPS/
├── venv/                    ✅ Environnement virtuel
├── data/
│   └── .gitkeep            ✅ Dossier prêt
├── models/
│   └── .gitkeep            ✅ Dossier prêt
├── results/
│   └── .gitkeep            ✅ Dossier prêt
├── src/                     ✅ Code source
│   ├── data_loader.py
│   ├── train.py
│   ├── hyperparameter_tuning.py
│   ├── evaluate.py
│   └── preprocessing.py
├── .github/workflows/       ✅ CI/CD
├── requirements.txt         ✅ Dépendances
├── dvc.yaml                ✅ Pipeline
└── DOCUMENTATION.md         ✅ Documentation
```

---

## ✅ Checklist Post-Installation

- [ ] Python 3.9+ installé
- [ ] Environnement virtuel créé et activé (`(venv)` visible)
- [ ] Dépendances installées (pas d'erreurs)
- [ ] Test de vérification réussi
- [ ] Premier dataset créé (V1)
- [ ] Premier modèle entraîné
- [ ] MLflow UI accessible (localhost:5000)

---

## 🚀 Prochaines Étapes

1. **Suivre le Guide d'Exécution**:
   ```powershell
   # Ouvrir dans VS Code ou votre éditeur
   code GUIDE_EXECUTION.md
   ```

2. **Créer les 3 Versions de Datasets**:
   ```powershell
   python change_dataset.py --all
   ```

3. **Entraîner Plusieurs Modèles**:
   ```powershell
   python src/train.py --data_path data/v2_filtered_housing.csv --model xgboost --data_version v2
   python src/train.py --data_path data/v3_engineered_housing.csv --model xgboost --data_version v3
   ```

4. **Optimiser avec Optuna**:
   ```powershell
   python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv --model xgboost --n_trials 50 --data_version v3
   ```

5. **Comparer les Résultats**:
   ```powershell
   python src/evaluate.py --compare_all
   mlflow ui
   ```

---

## 📚 Documentation

| Fichier | Quand l'Utiliser |
|---------|------------------|
| `QUICK_REFERENCE.md` | Commandes rapides |
| `GUIDE_EXECUTION.md` | Guide pas-à-pas complet |
| `DOCUMENTATION.md` | Référence complète du projet |
| `PROJET_RESUME.md` | Vue d'ensemble et résumé |

---

## 💡 Conseils

1. **Toujours activer l'environnement** avant de travailler
2. **MLflow UI** doit rester ouvert pendant l'entraînement pour voir les résultats en temps réel
3. **Commit réguliers** dans Git pour sauvegarder votre travail
4. **Vérifier les logs** en cas d'erreur (très détaillés)

---

## 🎉 Vous êtes Prêt!

Tout est configuré et prêt à être utilisé. Commencez par:

```powershell
# Workflow complet automatique
python run_complete_workflow.py

# Puis visualisez
mlflow ui
```

**Bonne chance avec votre projet MLOps! 🚀**
