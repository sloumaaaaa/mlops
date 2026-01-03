# 🚀 Installation et Premiers Pas

## ⚡ Installation Rapide (5 minutes)

### 1️⃣ Vérifier Python
```powershell
python --version
# Doit afficher Python 3.9 ou supérieur
```

### 2️⃣ Créer l'Environnement Virtuel
```powershell
cd C:\Users\sziedi\Desktop\esprit\mlops\MLOPS
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Note**: Si erreur "execution policy":
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3️⃣ Installer les Dépendances
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Temps estimé**: 3-5 minutes

### 4️⃣ Vérifier l'Installation
```powershell
python -c "import sklearn, mlflow, dvc, optuna, pandas, numpy; print('✅ Toutes les dépendances sont installées!')"
```

---

## 🎯 Premier Test (2 minutes)

### Créer le Premier Dataset
```powershell
python src/data_loader.py --version 1
```

**Résultat attendu**:
```
INFO - Loading California Housing dataset...
INFO - Dataset loaded: 20640 rows, 9 columns
INFO - Creating Version 1: Original dataset
INFO - Version 1 saved to data/v1_california_housing.csv
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
  RMSE: 0.4059
  MAE: 0.2925
  R²: 0.9031
  MAPE: 11.72%
✅ Model saved!
```

### Visualiser dans MLflow
```powershell
python -m mlflow ui --port 5000
```

Puis ouvrez: **http://127.0.0.1:5000**

---

## 📋 Workflow Complet Automatisé

### Option la Plus Simple
```powershell
python run_complete_workflow.py
```

Ce script va:
1. ✅ Créer les 3 versions de datasets
2. ✅ Entraîner plusieurs modèles
3. ✅ Générer les rapports de comparaison

**Temps estimé**: 5-10 minutes

---

## 🔧 Commandes Utiles

### Créer Tous les Datasets
```powershell
python src/data_loader.py --version 1
python src/data_loader.py --version 2
python src/data_loader.py --version 3
```

### Entraîner Modèles
```powershell
# Random Forest V1
python src/train.py --data_path data/v1_california_housing.csv --model random_forest --data_version v1

# Gradient Boosting V2
python src/train.py --data_path data/v2_filtered_housing.csv --model gradient_boosting --data_version v2

# Gradient Boosting V3
python src/train.py --data_path data/v3_engineered_housing.csv --model gradient_boosting --data_version v3
```

### Comparer Modèles
```powershell
python src/evaluate.py --compare_all
```

### Git
```powershell
git status
git add .
git commit -m "Update project"
git push origin dev
```

---

## 📊 Résultats Attendus

**Datasets créés:**
- v1_california_housing.csv (20,640 lignes)
- v2_filtered_housing.csv (10,297 lignes)
- v3_engineered_housing.csv (10,297×13)

**Modèles entraînés:**
- RandomForest V1: RMSE 0.4059
- GradientBoosting V2: RMSE 0.4043
- GradientBoosting V3: RMSE 0.4023

**Fichiers générés:**
- models/*.pkl
- results/model_comparison.txt
- results/model_comparison.png
- mlflow.db (base de données MLflow)

---

**Projet MLOps - ESPRIT - Janvier 2026**
