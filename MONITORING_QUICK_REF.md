# 📋 Monitoring Quick Reference

## 🎯 Commandes Essentielles

### Démonstration Complète
```bash
# Exécuter la démo avec données simulées
python demo_monitoring.py
```

### Configuration Initiale (Une Fois)
```bash
# Calculer les statistiques du dataset d'entraînement
python calculate_train_stats.py \
  --train-data data/v3_engineered_housing.csv \
  --target-column MedHouseVal \
  --threshold-multiplier 2.0
```

### Vérification Quotidienne/Hebdomadaire
```bash
# Vérifier le drift sur nouvelles données
python check_production_drift.py \
  --prod-data data/production_data.csv \
  --save-history

# Avec alerte pour CI/CD (exit code 1 si drift détecté)
python check_production_drift.py \
  --prod-data data/production_data.csv \
  --save-history \
  --alert-on-drift
```

## 🐍 Utilisation Programmatique

### Exemple Minimal
```python
from src.monitoring import DataDriftMonitor
import pandas as pd

# Initialiser
monitor = DataDriftMonitor()

# Sauvegarder stats d'entraînement (une fois)
df_train = pd.read_csv("data/train.csv")
monitor.save_train_statistics(df_train, target_column="target")

# Vérifier drift (régulièrement)
df_prod = pd.read_csv("data/prod.csv")
drift_results = monitor.detect_drift(df_prod, target_column="target")
monitor.print_drift_report(drift_results)
monitor.save_drift_history(drift_results)
```

### Monitoring des Métriques
```python
from src.monitoring import monitor_model_metrics

baseline_metrics = {"r2_score": 0.85, "rmse": 0.45}
current_metrics = {"r2_score": 0.78, "rmse": 0.52}

results = monitor_model_metrics(
    metrics=current_metrics,
    baseline_metrics=baseline_metrics,
    threshold_percentage=10.0
)

if results["degradation_detected"]:
    print("⚠️ Dégradation détectée - Réentraînement recommandé")
```

## 📊 Fichiers Générés

```
artifacts/
├── train_stats.json        # Statistiques d'entraînement (baseline)
└── drift_history.json      # Historique des vérifications

data/
├── simulated_prod_no_drift.csv    # (Démo) Données sans drift
└── simulated_prod_with_drift.csv  # (Démo) Données avec drift
```

## ⚙️ Paramètres Importants

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--threshold-multiplier` | 2.0 | Seuil drift = N × std |
| `--save-history` | False | Sauvegarder dans historique |
| `--alert-on-drift` | False | Exit code 1 si drift |
| Categorical threshold | 0.2 | Seuil features catégorielles |

## 🔔 Interprétation des Alertes

| Status | Description | Action |
|--------|-------------|--------|
| ✅ OK | Aucun drift | Continuer monitoring |
| ⚠️ DRIFT | Drift détecté | Investiguer cause |
| 🔴 ALERTE | >50% features driftées | Réentraînement urgent |

## 📖 Documentation Complète

- [MONITORING_GUIDE.md](MONITORING_GUIDE.md) - Guide complet
- [src/monitoring.py](src/monitoring.py) - Code source

---
💡 **Tip**: Commencez par `python demo_monitoring.py` pour voir le système en action!
