# Guide de Monitoring et Détection de Data Drift

## 📊 Vue d'ensemble

Ce système de monitoring permet de détecter le **data drift** dans vos données de production et de surveiller les performances de votre modèle de Machine Learning.

### Qu'est-ce que le Data Drift?

Le data drift survient lorsque les caractéristiques statistiques des données de production diffèrent significativement de celles utilisées pour entraîner le modèle. Cela peut entraîner une dégradation des performances du modèle.

## 🎯 Objectifs du système de monitoring

1. ✅ Calculer les statistiques du dataset d'entraînement
2. ✅ Sauvegarder ces statistiques pour référence
3. ✅ Comparer les données de production avec la baseline
4. ✅ Détecter un éventuel data drift
5. ✅ Monitorer les features et métriques
6. ✅ Garder un historique des vérifications
7. ✅ Définir des seuils clairs
8. ✅ Automatiser les alertes

## 🚀 Workflow complet

### Étape 1: Calculer les statistiques d'entraînement

Après avoir entraîné votre modèle, calculez et sauvegardez les statistiques du dataset d'entraînement:

```bash
python calculate_train_stats.py --train-data data/v3_engineered_housing.csv --target-column MedHouseVal
```

**Options disponibles:**
- `--train-data`: Chemin vers le dataset d'entraînement (défaut: `data/v3_engineered_housing.csv`)
- `--target-column`: Nom de la colonne cible à exclure (défaut: `MedHouseVal`)
- `--stats-path`: Où sauvegarder les stats (défaut: `artifacts/train_stats.json`)
- `--threshold-multiplier`: Multiplicateur du std pour le seuil (défaut: `2.0`)

**Sortie:**
- Fichier `artifacts/train_stats.json` contenant:
  - Moyennes, écarts-types, min, max, médiane des features numériques
  - Distributions des features catégorielles
  - Timestamp et métadonnées

### Étape 2: Vérifier le drift sur les données de production

Lorsque vous recevez de nouvelles données de production, vérifiez s'il y a du drift:

```bash
python check_production_drift.py --prod-data data/new_production_data.csv --save-history
```

**Options disponibles:**
- `--prod-data`: Chemin vers les données de production (**requis**)
- `--target-column`: Colonne cible à exclure (optionnel)
- `--stats-path`: Chemin vers les stats d'entraînement (défaut: `artifacts/train_stats.json`)
- `--save-history`: Sauvegarder dans l'historique
- `--history-path`: Chemin de l'historique (défaut: `artifacts/drift_history.json`)
- `--alert-on-drift`: Retourner code d'erreur si drift détecté (utile pour CI/CD)

**Sortie:**
- Rapport détaillé dans la console
- Historique sauvegardé (si `--save-history`)
- Code de sortie 1 si drift détecté (si `--alert-on-drift`)

## 📝 Utilisation programmatique

### Exemple basique

```python
from src.monitoring import DataDriftMonitor
import pandas as pd

# Initialiser le moniteur
monitor = DataDriftMonitor(
    stats_path="artifacts/train_stats.json",
    threshold_multiplier=2.0
)

# 1. Calculer et sauvegarder les stats d'entraînement
df_train = pd.read_csv("data/train.csv")
monitor.save_train_statistics(df_train, target_column="target")

# 2. Vérifier le drift sur les données de production
df_prod = pd.read_csv("data/prod.csv")
drift_results = monitor.detect_drift(df_prod, target_column="target")

# 3. Afficher le rapport
monitor.print_drift_report(drift_results)

# 4. Sauvegarder dans l'historique
monitor.save_drift_history(drift_results)
```

### Monitoring des métriques du modèle

```python
from src.monitoring import monitor_model_metrics

# Métriques de référence (baseline)
baseline_metrics = {
    "r2_score": 0.85,
    "rmse": 0.45,
    "mae": 0.32
}

# Métriques actuelles
current_metrics = {
    "r2_score": 0.78,  # Dégradation!
    "rmse": 0.52,
    "mae": 0.38
}

# Détecter la dégradation
results = monitor_model_metrics(
    metrics=current_metrics,
    baseline_metrics=baseline_metrics,
    threshold_percentage=10.0  # Alerte si dégradation > 10%
)

if results["degradation_detected"]:
    print("⚠️ Dégradation détectée!")
    for metric_info in results["metrics_degradation"]:
        if metric_info["degradation_detected"]:
            print(f"  - {metric_info['metric']}: {metric_info['percentage_change']:.2f}%")
```

## 🔍 Méthodes de détection

### Features numériques

Pour chaque feature numérique, le système:
1. Calcule la moyenne des données de production
2. Compare avec la moyenne d'entraînement
3. Détecte un drift si: `|mean_prod - mean_train| > threshold × std_train`

**Score de drift:** `difference / std_train` (normalisation)

### Features catégorielles

Pour chaque feature catégorielle, le système:
1. Calcule les distributions de probabilité
2. Mesure la divergence (distance de variation totale)
3. Détecte un drift si: `divergence > 0.2`
4. Identifie les nouvelles/manquantes catégories

## ⚙️ Configuration des seuils

### Threshold Multiplier (features numériques)

- **Valeur par défaut:** 2.0
- **Sensible (1.0-1.5):** Détecte des changements plus subtils
- **Standard (2.0-2.5):** Équilibre entre sensibilité et faux positifs
- **Conservatif (3.0+):** Ne détecte que les changements majeurs

### Threshold Categorical (features catégorielles)

- **Valeur par défaut:** 0.2 (20% de divergence)
- Ajustable dans le code si nécessaire

## 📈 Interprétation des résultats

### Rapport de drift

```
======================================================================
RAPPORT DE DÉTECTION DE DATA DRIFT
======================================================================
Timestamp: 2026-01-14T10:30:00
Échantillons production: 1000
Échantillons entraînement: 15000
Seuil (multiplicateur std): 2.0
----------------------------------------------------------------------

📊 FEATURES NUMÉRIQUES:

  ✓ OK - feature1
    Moyenne prod:  5.2345
    Moyenne train: 5.1234
    Différence:    0.1111 (seuil: 0.5000)
    Score drift:   0.22

  ⚠️  DRIFT DÉTECTÉ - feature2
    Moyenne prod:  10.5000
    Moyenne train: 8.2000
    Différence:    2.3000 (seuil: 1.0000)
    Score drift:   2.30

======================================================================
RÉSUMÉ:
  Total features:     10
  Features driftées:  3
  Pourcentage drift:  30.0%

  🔴 ALERTE: DATA DRIFT DÉTECTÉ!
  Action recommandée: Considérer le réentraînement du modèle
======================================================================
```

### Actions recommandées selon le drift

| % Features driftées | Statut | Action |
|---------------------|--------|--------|
| 0% | 🟢 Excellent | Aucune action requise |
| 1-20% | 🟡 Acceptable | Surveillance accrue |
| 21-50% | 🟠 Attention | Analyse approfondie, considérer réentraînement |
| >50% | 🔴 Critique | Réentraînement urgent recommandé |

## 📂 Structure des fichiers générés

### train_stats.json
```json
{
    "timestamp": "2026-01-14T10:00:00",
    "n_samples": 15000,
    "n_features": 8,
    "numeric_features": {
        "columns": ["feature1", "feature2", ...],
        "mean": {"feature1": 5.123, ...},
        "std": {"feature1": 0.456, ...},
        "min": {...},
        "max": {...},
        "median": {...}
    },
    "categorical_features": {
        "columns": ["category1", ...],
        "value_counts": {...}
    },
    "target_column": "target",
    "threshold_multiplier": 2.0
}
```

### drift_history.json
```json
{
    "checks": [
        {
            "timestamp": "2026-01-14T11:00:00",
            "drift_detected": true,
            "n_samples_prod": 1000,
            "numeric_drift": [...],
            "categorical_drift": [...],
            "summary": {
                "total_features": 10,
                "drifted_features": 3,
                "drift_percentage": 30.0
            }
        }
    ]
}
```

## 🔄 Automatisation avec CI/CD

### GitHub Actions / GitLab CI

```yaml
# .github/workflows/drift-monitoring.yml
name: Data Drift Monitoring

on:
  schedule:
    - cron: '0 */6 * * *'  # Toutes les 6 heures
  workflow_dispatch:

jobs:
  check-drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Check for drift
        run: |
          python check_production_drift.py \
            --prod-data data/latest_production.csv \
            --save-history \
            --alert-on-drift
      
      - name: Upload results
        if: failure()
        uses: actions/upload-artifact@v2
        with:
          name: drift-report
          path: artifacts/drift_history.json
```

## 🛠️ Intégration avec DVC

Ajoutez le monitoring à votre pipeline DVC:

```yaml
# dvc.yaml
stages:
  # ... autres stages ...
  
  calculate_train_stats:
    cmd: python calculate_train_stats.py
    deps:
      - data/v3_engineered_housing.csv
      - src/monitoring.py
    outs:
      - artifacts/train_stats.json
  
  check_drift:
    cmd: python check_production_drift.py --prod-data data/production.csv --save-history
    deps:
      - data/production.csv
      - artifacts/train_stats.json
      - src/monitoring.py
    metrics:
      - artifacts/drift_history.json
```

## 🎓 Bonnes pratiques

1. **Fréquence de vérification:**
   - Batch: Quotidien ou hebdomadaire
   - Streaming: Toutes les heures ou en temps réel

2. **Taille des échantillons:**
   - Minimum recommandé: 100 échantillons pour production
   - Idéal: >1000 échantillons pour statistiques robustes

3. **Action sur drift détecté:**
   - Investiguer les causes (changement métier, bug, saison...)
   - Collecter plus de données si nécessaire
   - Réentraîner le modèle avec données récentes
   - Mettre à jour les statistiques de référence

4. **Versioning:**
   - Versionner les statistiques avec Git
   - Lier aux versions du modèle (MLflow)
   - Documenter les changements de seuils

5. **Alertes:**
   - Configurer des notifications (email, Slack, Teams)
   - Définir des niveaux d'alerte (warning, critical)
   - Documenter les procédures d'escalade

## 🔗 Références

- [Documentation complète du projet](README.md)
- [Guide d'exécution](GUIDE_EXECUTION.md)
- [Module de monitoring](src/monitoring.py)
- [Calcul des stats](calculate_train_stats.py)
- [Vérification drift](check_production_drift.py)

---
📝 **Note:** Ce système de monitoring est évolutif. Vous pouvez ajouter d'autres métriques, 
intégrations, ou méthodes de détection selon vos besoins spécifiques.
