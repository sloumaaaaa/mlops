# California Housing Price Prediction - MLOps Project

[![ML Pipeline CI/CD](https://github.com/yourusername/mlops/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/yourusername/mlops/actions)

Un projet MLOps complet démontrant les meilleures pratiques pour le développement, le versioning, et le déploiement de modèles de Machine Learning.

## 🎯 Objectif du Projet

Prédire les prix de l'immobilier en Californie en utilisant un pipeline MLOps professionnel avec:
- ✅ **Git** pour le versioning du code
- ✅ **MLflow** pour le tracking des expériences
- ✅ **DVC** pour le versioning des données
- ✅ **GitHub Actions** pour l'automatisation CI/CD
- ✅ **Optuna** pour l'optimisation d'hyperparamètres (fonctionnalité avancée)

## 📊 Dataset

**California Housing Prices Dataset**
- 20,640 échantillons
- 8 features (MedInc, HouseAge, AveRooms, etc.)
- Target: Prix médian des maisons

**3 Versions du Dataset:**
1. **V1**: Dataset original complet
2. **V2**: Dataset filtré (outliers supprimés, focus côtier)
3. **V3**: Feature engineering (4 nouvelles features créées)

## 🚀 Installation Rapide

```bash
# Cloner le repository
git clone <your-repo-url>
cd MLOPS

# Créer environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Initialiser DVC
dvc init
```

## 🏃 Quick Start

### Option 1: Workflow Automatisé Complet
```bash
python run_complete_workflow.py
```

Cette commande exécute automatiquement:
1. Création des 3 versions de datasets
2. Entraînement de plusieurs modèles
3. Optimisation d'hyperparamètres avec Optuna
4. Évaluation et comparaison des résultats
- Paramètres utilisés (`n_estimators`, `random_state`)
- Modèles sauvegardés (section "Artifacts" -> `model`)

Explication du code (`train_mlflow.py`):
- On fixe l'experiment avec `mlflow.set_experiment("iris-mlops")` pour regrouper les runs.
- Pour chaque exécution on ouvre un `mlflow.start_run()` qui crée un run isolé.
- Le modèle `RandomForestClassifier` est entraîné sur Iris, puis on calcule `accuracy` et `precision`.
- On loggue les paramètres (`mlflow.log_param`) et les métriques (`mlflow.log_metric`).
- On sauvegarde le modèle avec `mlflow.sklearn.log_model` (stocké comme artifact du run).

Modifications demandées (exemples):
- Changer `n_estimators` en passant `--n_estimators` à `train_mlflow.py`.
- Changer `random_state` en passant `--random_state`.
- Ajouter la métrique `precision` (déjà implémentée dans `train_mlflow.py`):

```python
from sklearn.metrics import precision_score
mlflow.log_metric("precision", precision_score(y_test, preds, average="macro"))
```

Questions / Réponses:

- Pourquoi MLflow est-il indispensable en MLOps ?
  MLflow fournit un tracking centralisé des expériences, paramètres, métriques et modèles. Il facilite la reproductibilité, la comparaison de runs, la gestion des artefacts et l'intégration dans des pipelines CI/CD et de déploiement. En MLOps, le suivi systématique et la traçabilité sont essentiels; MLflow automatise et standardise ces aspects.

- Quelle différence entre un `run` et un `experiment` ?
  Un `experiment` est un conteneur logique rassemblant plusieurs `runs`. Un `run` correspond à une exécution unique (une tentative d'entraînement) avec ses paramètres, métriques et artefacts; un `experiment` permet d'organiser et comparer plusieurs runs du même projet.

- Peut-on reproduire un modèle sans tracking ?
  Théoriquement oui si vous conservez manuellement le code, la seed, les données et les hyperparamètres. En pratique, sans tracking il est facile d'oublier des détails (versions, preprocessings, seeds) rendant la reproduction difficile. Le tracking réduit ces risques.

---

Si vous voulez, je peux:
- Installer les dépendances et lancer une exécution de `train_mlflow.py` ici.
- Lancer `mlflow ui` pour vous (si vous confirmez).

Dites quelle action je dois faire ensuite.
