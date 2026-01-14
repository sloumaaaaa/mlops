"""
Script de démonstration du système de monitoring
Simule des données de production avec et sans drift pour tester le système
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.monitoring import DataDriftMonitor


def create_production_data_with_drift(base_data_path: str, output_path: str, 
                                      drift_features: list = None, drift_magnitude: float = 3.0):
    """
    Crée des données de production simulées avec du drift
    
    Args:
        base_data_path: Chemin vers les données de base
        output_path: Où sauvegarder les données simulées
        drift_features: Liste des features à modifier (None = toutes)
        drift_magnitude: Magnitude du drift (multiplicateur de std)
    """
    print(f"📊 Création de données de production avec drift...")
    
    # Charger les données de base
    df = pd.read_csv(base_data_path)
    
    # Prendre un échantillon pour la production
    df_prod = df.sample(n=min(1000, len(df)), random_state=42).copy()
    
    # Ajouter du drift sur les features numériques
    numeric_cols = df_prod.select_dtypes(include=[np.number]).columns.tolist()
    
    if drift_features is None:
        # Appliquer du drift sur 30% des features aléatoirement
        n_drift = max(1, int(len(numeric_cols) * 0.3))
        drift_features = np.random.choice(numeric_cols, n_drift, replace=False)
    
    print(f"   Features avec drift: {list(drift_features)}")
    
    for col in drift_features:
        if col in numeric_cols:
            # Ajouter un shift basé sur l'écart-type
            shift = drift_magnitude * df_prod[col].std()
            df_prod[col] = df_prod[col] + shift
            print(f"   - {col}: shift de {shift:.4f}")
    
    # Sauvegarder
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_prod.to_csv(output_path, index=False)
    print(f"✅ Données sauvegardées: {output_path}")
    
    return df_prod


def create_production_data_no_drift(base_data_path: str, output_path: str):
    """
    Crée des données de production simulées sans drift
    
    Args:
        base_data_path: Chemin vers les données de base
        output_path: Où sauvegarder les données simulées
    """
    print(f"📊 Création de données de production sans drift...")
    
    # Charger les données de base
    df = pd.read_csv(base_data_path)
    
    # Prendre un échantillon différent
    df_prod = df.sample(n=min(1000, len(df)), random_state=123).copy()
    
    # Sauvegarder
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_prod.to_csv(output_path, index=False)
    print(f"✅ Données sauvegardées: {output_path}")
    
    return df_prod


def run_demo():
    """
    Exécute une démonstration complète du système de monitoring
    """
    print("="*70)
    print("DÉMONSTRATION DU SYSTÈME DE MONITORING")
    print("="*70)
    
    # Configuration
    base_data = "data/v3_engineered_housing.csv"
    target_column = "MedHouseVal"
    
    # Vérifier que les données de base existent
    if not Path(base_data).exists():
        print(f"\n❌ Fichier de base non trouvé: {base_data}")
        print("Fichiers disponibles:")
        for f in Path("data").glob("*.csv"):
            print(f"  - {f}")
        return
    
    print(f"\n📁 Dataset de base: {base_data}")
    df_base = pd.read_csv(base_data)
    print(f"   Shape: {df_base.shape}")
    
    # Étape 1: Calculer les statistiques d'entraînement
    print("\n" + "="*70)
    print("ÉTAPE 1: Calcul des statistiques d'entraînement")
    print("="*70)
    
    monitor = DataDriftMonitor(
        stats_path="artifacts/train_stats.json",
        history_path="artifacts/drift_history.json",
        threshold_multiplier=2.0
    )
    
    monitor.save_train_statistics(df_base, target_column=target_column)
    
    # Étape 2: Créer des données de production SANS drift
    print("\n" + "="*70)
    print("ÉTAPE 2: Test avec données SANS drift")
    print("="*70)
    
    df_prod_no_drift = create_production_data_no_drift(
        base_data, 
        "data/simulated_prod_no_drift.csv"
    )
    
    print("\n🔍 Détection du drift...")
    drift_results_no = monitor.detect_drift(df_prod_no_drift, target_column=target_column)
    monitor.print_drift_report(drift_results_no)
    monitor.save_drift_history(drift_results_no)
    
    # Étape 3: Créer des données de production AVEC drift
    print("\n" + "="*70)
    print("ÉTAPE 3: Test avec données AVEC drift")
    print("="*70)
    
    # Sélectionner quelques features pour le drift
    numeric_cols = df_base.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    
    drift_features = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols[:1]
    
    df_prod_with_drift = create_production_data_with_drift(
        base_data,
        "data/simulated_prod_with_drift.csv",
        drift_features=drift_features,
        drift_magnitude=3.0
    )
    
    print("\n🔍 Détection du drift...")
    drift_results_yes = monitor.detect_drift(df_prod_with_drift, target_column=target_column)
    monitor.print_drift_report(drift_results_yes)
    monitor.save_drift_history(drift_results_yes)
    
    # Étape 4: Afficher l'historique
    print("\n" + "="*70)
    print("ÉTAPE 4: Historique des vérifications")
    print("="*70)
    
    history_summary = monitor.get_drift_history_summary()
    print(f"\n📊 Résumé de l'historique:")
    print(f"   Total vérifications: {history_summary.get('total_checks', 0)}")
    print(f"   Drift détecté: {history_summary.get('drift_detected_count', 0)} fois")
    print(f"   Taux de drift: {history_summary.get('drift_rate', 0):.1f}%")
    
    # Résumé final
    print("\n" + "="*70)
    print("RÉSUMÉ DE LA DÉMONSTRATION")
    print("="*70)
    print("\n✅ Fichiers créés:")
    print(f"   - {monitor.stats_path}")
    print(f"   - {monitor.history_path}")
    print(f"   - data/simulated_prod_no_drift.csv")
    print(f"   - data/simulated_prod_with_drift.csv")
    
    print("\n📝 Résultats:")
    print(f"   - Test sans drift: {'✅ AUCUN drift détecté' if not drift_results_no['drift_detected'] else '❌ Drift détecté (inattendu)'}")
    print(f"   - Test avec drift: {'✅ DRIFT détecté' if drift_results_yes['drift_detected'] else '❌ Aucun drift (inattendu)'}")
    
    print("\n🎯 Prochaines étapes:")
    print("   1. Examiner les rapports ci-dessus")
    print("   2. Consulter artifacts/drift_history.json")
    print("   3. Tester avec vos propres données:")
    print("      python check_production_drift.py --prod-data <votre_fichier.csv> --save-history")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    run_demo()
