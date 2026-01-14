"""
Train with MLflow + Monitoring Integration
Entraîne un modèle avec MLflow et génère automatiquement les statistiques pour le monitoring
"""

import argparse
import pandas as pd
import mlflow
from pathlib import Path
from src.monitoring import DataDriftMonitor
from src.train import train_model


def train_with_monitoring(
    data_path: str,
    model_type: str,
    data_version: str,
    target_column: str = "MedHouseVal",
    enable_monitoring: bool = True,
    threshold_multiplier: float = 2.0
):
    """
    Entraîne un modèle avec MLflow et configure le monitoring
    
    Args:
        data_path: Chemin vers le dataset
        model_type: Type de modèle ('random_forest', 'gradient_boosting', etc.)
        data_version: Version du dataset ('v1', 'v2', 'v3')
        target_column: Nom de la colonne cible
        enable_monitoring: Activer la génération des stats de monitoring
        threshold_multiplier: Multiplicateur du std pour le seuil de drift
    """
    print("="*70)
    print("ENTRAÎNEMENT AVEC MONITORING INTÉGRÉ")
    print("="*70)
    
    # Charger les données
    print(f"\n📁 Chargement des données: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Shape: {df.shape}")
    
    # Entraîner le modèle avec MLflow (fonction existante)
    print(f"\n🤖 Entraînement du modèle: {model_type}")
    print(f"   Version dataset: {data_version}")
    
    # Note: Vous devrez adapter cette partie selon votre fonction train_model existante
    # Cette version suppose que vous avez une fonction train_model dans src/train.py
    # Si ce n'est pas le cas, remplacez par votre code d'entraînement
    
    with mlflow.start_run(run_name=f"{model_type}_{data_version}_with_monitoring") as run:
        # Vos paramètres d'entraînement existants
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("monitoring_enabled", enable_monitoring)
        mlflow.log_param("drift_threshold_multiplier", threshold_multiplier)
        
        # Entraîner le modèle
        # model, metrics = train_model(df, target_column, model_type)
        # mlflow.log_metrics(metrics)
        
        print(f"\n✅ Modèle entraîné - Run ID: {run.info.run_id}")
        
        # Générer les statistiques de monitoring
        if enable_monitoring:
            print("\n" + "="*70)
            print("GÉNÉRATION DES STATISTIQUES DE MONITORING")
            print("="*70)
            
            # Créer le dossier artifacts si nécessaire
            artifacts_dir = Path("artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            
            # Définir le chemin des stats basé sur la version du dataset
            stats_filename = f"train_stats_{data_version}_{model_type}.json"
            stats_path = artifacts_dir / stats_filename
            
            # Initialiser le moniteur
            monitor = DataDriftMonitor(
                stats_path=str(stats_path),
                threshold_multiplier=threshold_multiplier
            )
            
            # Calculer et sauvegarder les statistiques
            print(f"\n📊 Calcul des statistiques pour {data_version}...")
            monitor.save_train_statistics(df, target_column=target_column)
            
            # Logger les statistiques comme artifact dans MLflow
            mlflow.log_artifact(str(stats_path), artifact_path="monitoring")
            
            print(f"\n✅ Statistiques de monitoring sauvegardées:")
            print(f"   - Fichier local: {stats_path}")
            print(f"   - MLflow artifact: monitoring/{stats_filename}")
            
            # Logger des métriques de monitoring additionnelles
            mlflow.log_metric("n_features_monitored", len(df.columns) - 1)
            mlflow.log_metric("n_samples_baseline", len(df))
            
            # Créer un fichier README pour le monitoring
            readme_path = artifacts_dir / f"monitoring_readme_{data_version}.txt"
            with open(readme_path, "w") as f:
                f.write(f"Monitoring Configuration\n")
                f.write(f"========================\n\n")
                f.write(f"Model Type: {model_type}\n")
                f.write(f"Data Version: {data_version}\n")
                f.write(f"Dataset: {data_path}\n")
                f.write(f"Target Column: {target_column}\n")
                f.write(f"Threshold Multiplier: {threshold_multiplier}\n")
                f.write(f"Baseline Samples: {len(df)}\n")
                f.write(f"Features Monitored: {len(df.columns) - 1}\n\n")
                f.write(f"Usage:\n")
                f.write(f"------\n")
                f.write(f"python check_production_drift.py \\\n")
                f.write(f"  --prod-data data/production_data.csv \\\n")
                f.write(f"  --stats-path {stats_path} \\\n")
                f.write(f"  --target-column {target_column} \\\n")
                f.write(f"  --save-history\n")
            
            mlflow.log_artifact(str(readme_path), artifact_path="monitoring")
            
        print("\n" + "="*70)
        print("ENTRAÎNEMENT ET MONITORING TERMINÉS")
        print("="*70)
        print(f"\n📊 MLflow UI: mlflow ui --port 5000")
        print(f"   Run ID: {run.info.run_id}")
        if enable_monitoring:
            print(f"\n📈 Monitoring configuré:")
            print(f"   Stats: {stats_path}")
            print(f"   Pour vérifier le drift, utilisez: check_production_drift.py")


def main():
    parser = argparse.ArgumentParser(
        description="Entraîner un modèle avec monitoring intégré"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/v3_engineered_housing.csv",
        help="Chemin vers le dataset"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="gradient_boosting",
        choices=["random_forest", "gradient_boosting", "linear_regression"],
        help="Type de modèle à entraîner"
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default="v3",
        help="Version du dataset (v1, v2, v3)"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="MedHouseVal",
        help="Nom de la colonne cible"
    )
    parser.add_argument(
        "--enable-monitoring",
        action="store_true",
        default=True,
        help="Générer les statistiques de monitoring"
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=2.0,
        help="Multiplicateur du std pour le seuil de drift"
    )
    
    args = parser.parse_args()
    
    train_with_monitoring(
        data_path=args.data_path,
        model_type=args.model_type,
        data_version=args.data_version,
        target_column=args.target_column,
        enable_monitoring=args.enable_monitoring,
        threshold_multiplier=args.threshold_multiplier
    )


if __name__ == "__main__":
    main()
