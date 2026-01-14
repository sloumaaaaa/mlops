"""
Script pour détecter le data drift dans les données de production
Compare les données de production avec les statistiques d'entraînement
"""

import pandas as pd
import argparse
from pathlib import Path
from src.monitoring import DataDriftMonitor
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Détecter le data drift dans les données de production"
    )
    parser.add_argument(
        "--prod-data",
        type=str,
        required=True,
        help="Chemin vers le dataset de production"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Nom de la colonne cible (sera exclue de l'analyse)"
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default="artifacts/train_stats.json",
        help="Chemin vers les statistiques d'entraînement"
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Sauvegarder les résultats dans l'historique"
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default="artifacts/drift_history.json",
        help="Chemin vers l'historique de drift"
    )
    parser.add_argument(
        "--alert-on-drift",
        action="store_true",
        help="Retourner un code d'erreur si drift détecté (utile pour CI/CD)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("VÉRIFICATION DU DATA DRIFT - DONNÉES DE PRODUCTION")
    print("="*70)
    
    # Vérifier que le fichier de production existe
    prod_path = Path(args.prod_data)
    if not prod_path.exists():
        print(f"❌ Erreur: Fichier de production non trouvé: {prod_path}")
        sys.exit(1)
    
    # Vérifier que les statistiques d'entraînement existent
    if not Path(args.stats_path).exists():
        print(f"❌ Erreur: Statistiques d'entraînement non trouvées: {args.stats_path}")
        print("\nVeuillez d'abord calculer les statistiques avec:")
        print("  python calculate_train_stats.py")
        sys.exit(1)
    
    # Charger les données de production
    print(f"\n📁 Chargement des données de production: {args.prod_data}")
    df_prod = pd.read_csv(args.prod_data)
    print(f"  - Shape: {df_prod.shape}")
    print(f"  - Colonnes: {list(df_prod.columns)}")
    
    # Créer le moniteur
    monitor = DataDriftMonitor(
        stats_path=args.stats_path,
        history_path=args.history_path
    )
    
    # Détecter le drift
    print(f"\n🔍 Détection du data drift en cours...")
    drift_results = monitor.detect_drift(df_prod, target_column=args.target_column)
    
    # Afficher le rapport
    monitor.print_drift_report(drift_results)
    
    # Sauvegarder l'historique si demandé
    if args.save_history:
        monitor.save_drift_history(drift_results)
    
    # Afficher l'historique
    print("\n📜 HISTORIQUE DES VÉRIFICATIONS:")
    history_summary = monitor.get_drift_history_summary()
    if history_summary.get("total_checks", 0) > 0:
        print(f"  Total vérifications: {history_summary['total_checks']}")
        print(f"  Drift détecté: {history_summary['drift_detected_count']} fois")
        print(f"  Taux de drift: {history_summary['drift_rate']:.1f}%")
        print(f"  Dernière vérification: {history_summary['last_check']}")
    else:
        print(f"  {history_summary.get('message', 'Aucun historique')}")
    
    # Retourner un code d'erreur si drift détecté et alerte activée
    if args.alert_on_drift and drift_results["drift_detected"]:
        print("\n⚠️  Sortie avec code d'erreur (drift détecté)")
        sys.exit(1)
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
