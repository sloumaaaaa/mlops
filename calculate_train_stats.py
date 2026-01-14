"""
Script pour calculer et sauvegarder les statistiques du dataset d'entraînement
À exécuter après l'entraînement du modèle pour établir la baseline de monitoring
"""

import pandas as pd
import argparse
from pathlib import Path
from src.monitoring import DataDriftMonitor


def main():
    parser = argparse.ArgumentParser(
        description="Calculer les statistiques du dataset d'entraînement pour le monitoring"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/v3_engineered_housing.csv",
        help="Chemin vers le dataset d'entraînement"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="MedHouseVal",
        help="Nom de la colonne cible (sera exclue des statistiques)"
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default="artifacts/train_stats.json",
        help="Chemin où sauvegarder les statistiques"
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=2.0,
        help="Multiplicateur du std pour le seuil de drift (par défaut 2.0)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("CALCUL DES STATISTIQUES D'ENTRAÎNEMENT")
    print("="*70)
    
    # Vérifier que le fichier existe
    train_path = Path(args.train_data)
    if not train_path.exists():
        print(f"❌ Erreur: Fichier non trouvé: {train_path}")
        print("\nFichiers disponibles dans data/:")
        data_dir = Path("data")
        if data_dir.exists():
            for file in data_dir.glob("*.csv"):
                print(f"  - {file}")
        return
    
    # Charger les données
    print(f"\n📁 Chargement des données depuis: {args.train_data}")
    df_train = pd.read_csv(args.train_data)
    print(f"  - Shape: {df_train.shape}")
    print(f"  - Colonnes: {list(df_train.columns)}")
    
    # Vérifier que la colonne cible existe
    if args.target_column and args.target_column not in df_train.columns:
        print(f"\n⚠️  Attention: La colonne cible '{args.target_column}' n'existe pas")
        print(f"   Colonnes disponibles: {list(df_train.columns)}")
        response = input("   Continuer sans exclure de colonne cible? (y/n): ")
        if response.lower() != 'y':
            return
        args.target_column = None
    
    # Créer le moniteur
    monitor = DataDriftMonitor(
        stats_path=args.stats_path,
        threshold_multiplier=args.threshold_multiplier
    )
    
    # Calculer et sauvegarder les statistiques
    print(f"\n📊 Calcul des statistiques...")
    monitor.save_train_statistics(df_train, target_column=args.target_column)
    
    print(f"\n✅ Statistiques sauvegardées avec succès!")
    print(f"   Fichier: {args.stats_path}")
    print(f"   Seuil de drift: {args.threshold_multiplier} × std")
    print("\nVous pouvez maintenant utiliser ces statistiques pour détecter le drift")
    print("avec le script: python check_production_drift.py")
    print("="*70)


if __name__ == "__main__":
    main()
