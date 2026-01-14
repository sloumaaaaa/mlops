"""
Module de monitoring pour la détection de data drift
Permet de surveiller les features, métriques, et détecter les dérives de données
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional


class DataDriftMonitor:
    """
    Classe pour surveiller le data drift entre les données d'entraînement et de production
    """
    
    def __init__(self, stats_path: str = "artifacts/train_stats.json",
                 history_path: str = "artifacts/drift_history.json",
                 threshold_multiplier: float = 2.0):
        """
        Initialise le moniteur de drift
        
        Args:
            stats_path: Chemin vers le fichier des statistiques d'entraînement
            history_path: Chemin vers l'historique des vérifications de drift
            threshold_multiplier: Multiplicateur du std pour le seuil de drift (par défaut 2.0)
        """
        self.stats_path = Path(stats_path)
        self.history_path = Path(history_path)
        self.threshold_multiplier = threshold_multiplier
        self.train_stats = None
        
        # Créer le dossier artifacts s'il n'existe pas
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
    
    def calculate_train_statistics(self, df_train: pd.DataFrame, 
                                   target_column: Optional[str] = None) -> Dict:
        """
        Calcule les statistiques du dataset d'entraînement
        
        Args:
            df_train: DataFrame d'entraînement
            target_column: Nom de la colonne cible à exclure (optionnel)
        
        Returns:
            Dictionnaire contenant les statistiques
        """
        # Exclure la colonne cible si spécifiée
        if target_column and target_column in df_train.columns:
            df_features = df_train.drop(columns=[target_column])
        else:
            df_features = df_train
        
        # Séparer les colonnes numériques et catégorielles
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(df_train),
            "n_features": len(df_features.columns),
            "numeric_features": {
                "columns": numeric_cols,
                "mean": df_features[numeric_cols].mean().to_dict() if numeric_cols else {},
                "std": df_features[numeric_cols].std().to_dict() if numeric_cols else {},
                "min": df_features[numeric_cols].min().to_dict() if numeric_cols else {},
                "max": df_features[numeric_cols].max().to_dict() if numeric_cols else {},
                "median": df_features[numeric_cols].median().to_dict() if numeric_cols else {}
            },
            "categorical_features": {
                "columns": categorical_cols,
                "value_counts": {col: df_features[col].value_counts().to_dict() 
                                for col in categorical_cols} if categorical_cols else {}
            },
            "target_column": target_column,
            "threshold_multiplier": self.threshold_multiplier
        }
        
        return stats
    
    def save_train_statistics(self, df_train: pd.DataFrame, 
                             target_column: Optional[str] = None):
        """
        Calcule et sauvegarde les statistiques d'entraînement
        
        Args:
            df_train: DataFrame d'entraînement
            target_column: Nom de la colonne cible à exclure (optionnel)
        """
        stats = self.calculate_train_statistics(df_train, target_column)
        
        with open(self.stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        
        print(f"✓ Statistiques d'entraînement sauvegardées dans {self.stats_path}")
        print(f"  - {stats['n_samples']} échantillons")
        print(f"  - {len(stats['numeric_features']['columns'])} features numériques")
        print(f"  - {len(stats['categorical_features']['columns'])} features catégorielles")
    
    def load_train_statistics(self) -> Dict:
        """
        Charge les statistiques d'entraînement depuis le fichier
        
        Returns:
            Dictionnaire des statistiques
        """
        if not self.stats_path.exists():
            raise FileNotFoundError(
                f"Fichier de statistiques non trouvé: {self.stats_path}\n"
                "Veuillez d'abord calculer les statistiques d'entraînement."
            )
        
        with open(self.stats_path, "r") as f:
            self.train_stats = json.load(f)
        
        return self.train_stats
    
    def detect_drift(self, df_prod: pd.DataFrame, 
                    target_column: Optional[str] = None) -> Dict:
        """
        Détecte le data drift entre les données d'entraînement et de production
        
        Args:
            df_prod: DataFrame de production
            target_column: Nom de la colonne cible à exclure (optionnel)
        
        Returns:
            Dictionnaire contenant les résultats de la détection
        """
        # Charger les stats d'entraînement si pas déjà fait
        if self.train_stats is None:
            self.load_train_statistics()
        
        # Exclure la colonne cible si spécifiée
        if target_column and target_column in df_prod.columns:
            df_prod = df_prod.drop(columns=[target_column])
        
        timestamp = datetime.now().isoformat()
        drift_results = {
            "timestamp": timestamp,
            "n_samples_prod": len(df_prod),
            "n_samples_train": self.train_stats["n_samples"],
            "threshold_multiplier": self.train_stats["threshold_multiplier"],
            "numeric_drift": [],
            "categorical_drift": [],
            "drift_detected": False,
            "summary": {}
        }
        
        # Détection de drift pour les features numériques
        numeric_cols = self.train_stats["numeric_features"]["columns"]
        for col in numeric_cols:
            if col not in df_prod.columns:
                warnings.warn(f"Feature '{col}' manquante dans les données de production")
                continue
            
            prod_mean = df_prod[col].mean()
            train_mean = self.train_stats["numeric_features"]["mean"][col]
            train_std = self.train_stats["numeric_features"]["std"][col]
            
            # Calcul de la différence absolue
            diff = abs(prod_mean - train_mean)
            threshold = self.threshold_multiplier * train_std
            
            drift_detected = diff > threshold
            
            drift_info = {
                "feature": col,
                "prod_mean": float(prod_mean),
                "train_mean": float(train_mean),
                "train_std": float(train_std),
                "difference": float(diff),
                "threshold": float(threshold),
                "drift_detected": bool(drift_detected),
                "drift_score": float(diff / (train_std + 1e-10))  # Normalisation
            }
            
            drift_results["numeric_drift"].append(drift_info)
            
            if drift_detected:
                drift_results["drift_detected"] = True
        
        # Détection de drift pour les features catégorielles
        categorical_cols = self.train_stats["categorical_features"]["columns"]
        for col in categorical_cols:
            if col not in df_prod.columns:
                warnings.warn(f"Feature '{col}' manquante dans les données de production")
                continue
            
            prod_dist = df_prod[col].value_counts(normalize=True).to_dict()
            train_dist = self.train_stats["categorical_features"]["value_counts"][col]
            
            # Normaliser la distribution d'entraînement
            total = sum(train_dist.values())
            train_dist_norm = {k: v/total for k, v in train_dist.items()}
            
            # Calculer la divergence (distance de variation totale)
            all_values = set(prod_dist.keys()) | set(train_dist_norm.keys())
            divergence = sum(abs(prod_dist.get(v, 0) - train_dist_norm.get(v, 0)) 
                           for v in all_values) / 2
            
            # Seuil de 0.2 pour les features catégorielles
            drift_detected = divergence > 0.2
            
            drift_info = {
                "feature": col,
                "divergence": float(divergence),
                "threshold": 0.2,
                "drift_detected": bool(drift_detected),
                "new_categories": list(set(prod_dist.keys()) - set(train_dist_norm.keys())),
                "missing_categories": list(set(train_dist_norm.keys()) - set(prod_dist.keys()))
            }
            
            drift_results["categorical_drift"].append(drift_info)
            
            if drift_detected:
                drift_results["drift_detected"] = True
        
        # Résumé
        total_features = len(numeric_cols) + len(categorical_cols)
        drifted_features = sum(1 for d in drift_results["numeric_drift"] if d["drift_detected"])
        drifted_features += sum(1 for d in drift_results["categorical_drift"] if d["drift_detected"])
        
        drift_results["summary"] = {
            "total_features": total_features,
            "drifted_features": drifted_features,
            "drift_percentage": (drifted_features / total_features * 100) if total_features > 0 else 0
        }
        
        return drift_results
    
    def print_drift_report(self, drift_results: Dict):
        """
        Affiche un rapport formaté des résultats de drift
        
        Args:
            drift_results: Résultats de la détection de drift
        """
        print("\n" + "="*70)
        print("RAPPORT DE DÉTECTION DE DATA DRIFT")
        print("="*70)
        print(f"Timestamp: {drift_results['timestamp']}")
        print(f"Échantillons production: {drift_results['n_samples_prod']}")
        print(f"Échantillons entraînement: {drift_results['n_samples_train']}")
        print(f"Seuil (multiplicateur std): {drift_results['threshold_multiplier']}")
        print("-"*70)
        
        # Features numériques
        if drift_results["numeric_drift"]:
            print("\n📊 FEATURES NUMÉRIQUES:")
            for drift_info in drift_results["numeric_drift"]:
                status = "⚠️  DRIFT DÉTECTÉ" if drift_info["drift_detected"] else "✓ OK"
                print(f"\n  {status} - {drift_info['feature']}")
                print(f"    Moyenne prod:  {drift_info['prod_mean']:.4f}")
                print(f"    Moyenne train: {drift_info['train_mean']:.4f}")
                print(f"    Différence:    {drift_info['difference']:.4f} (seuil: {drift_info['threshold']:.4f})")
                print(f"    Score drift:   {drift_info['drift_score']:.2f}")
        
        # Features catégorielles
        if drift_results["categorical_drift"]:
            print("\n🏷️  FEATURES CATÉGORIELLES:")
            for drift_info in drift_results["categorical_drift"]:
                status = "⚠️  DRIFT DÉTECTÉ" if drift_info["drift_detected"] else "✓ OK"
                print(f"\n  {status} - {drift_info['feature']}")
                print(f"    Divergence: {drift_info['divergence']:.4f} (seuil: {drift_info['threshold']})")
                if drift_info["new_categories"]:
                    print(f"    Nouvelles catégories: {drift_info['new_categories']}")
                if drift_info["missing_categories"]:
                    print(f"    Catégories manquantes: {drift_info['missing_categories']}")
        
        # Résumé
        print("\n" + "="*70)
        print("RÉSUMÉ:")
        print(f"  Total features:     {drift_results['summary']['total_features']}")
        print(f"  Features driftées:  {drift_results['summary']['drifted_features']}")
        print(f"  Pourcentage drift:  {drift_results['summary']['drift_percentage']:.1f}%")
        
        if drift_results["drift_detected"]:
            print("\n  🔴 ALERTE: DATA DRIFT DÉTECTÉ!")
            print("  Action recommandée: Considérer le réentraînement du modèle")
        else:
            print("\n  🟢 Aucun drift détecté - Modèle stable")
        
        print("="*70 + "\n")
    
    def save_drift_history(self, drift_results: Dict):
        """
        Sauvegarde les résultats dans l'historique
        
        Args:
            drift_results: Résultats de la détection de drift
        """
        # Charger l'historique existant
        if self.history_path.exists():
            try:
                with open(self.history_path, "r") as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                # Si le fichier est corrompu, le réinitialiser
                warnings.warn(f"Fichier historique corrompu, réinitialisation: {self.history_path}")
                history = {"checks": []}
        else:
            history = {"checks": []}
        
        # Ajouter la nouvelle vérification
        history["checks"].append(drift_results)
        
        # Sauvegarder
        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=4)
        
        print(f"✓ Historique sauvegardé dans {self.history_path}")
    
    def get_drift_history_summary(self) -> Dict:
        """
        Obtient un résumé de l'historique des vérifications
        
        Returns:
            Dictionnaire avec statistiques sur l'historique
        """
        if not self.history_path.exists():
            return {"total_checks": 0, "message": "Aucun historique disponible"}
        
        with open(self.history_path, "r") as f:
            history = json.load(f)
        
        checks = history.get("checks", [])
        
        if not checks:
            return {"total_checks": 0, "message": "Aucune vérification dans l'historique"}
        
        drift_detected_count = sum(1 for check in checks if check.get("drift_detected", False))
        
        summary = {
            "total_checks": len(checks),
            "drift_detected_count": drift_detected_count,
            "drift_rate": (drift_detected_count / len(checks) * 100) if checks else 0,
            "last_check": checks[-1]["timestamp"],
            "last_drift_detected": checks[-1]["drift_detected"]
        }
        
        return summary


def monitor_model_metrics(metrics: Dict[str, float], 
                         baseline_metrics: Dict[str, float],
                         threshold_percentage: float = 10.0) -> Dict:
    """
    Monitore les métriques du modèle et détecte les dégradations
    
    Args:
        metrics: Métriques actuelles du modèle
        baseline_metrics: Métriques de référence (baseline)
        threshold_percentage: Seuil de dégradation en pourcentage (par défaut 10%)
    
    Returns:
        Dictionnaire avec les résultats du monitoring
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "metrics_degradation": [],
        "degradation_detected": False
    }
    
    for metric_name, current_value in metrics.items():
        if metric_name not in baseline_metrics:
            continue
        
        baseline_value = baseline_metrics[metric_name]
        
        # Calculer la différence en pourcentage
        if baseline_value != 0:
            percentage_change = ((current_value - baseline_value) / abs(baseline_value)) * 100
        else:
            percentage_change = 0
        
        # Pour les métriques où plus c'est haut mieux c'est (accuracy, r2, etc.)
        # Une baisse est une dégradation
        degradation = abs(percentage_change) > threshold_percentage and percentage_change < 0
        
        metric_info = {
            "metric": metric_name,
            "current_value": current_value,
            "baseline_value": baseline_value,
            "percentage_change": percentage_change,
            "degradation_detected": degradation
        }
        
        results["metrics_degradation"].append(metric_info)
        
        if degradation:
            results["degradation_detected"] = True
    
    return results


if __name__ == "__main__":
    # Exemple d'utilisation
    print("Module de monitoring chargé avec succès!")
    print("\nUtilisation:")
    print("  from src.monitoring import DataDriftMonitor")
    print("  monitor = DataDriftMonitor()")
    print("  monitor.save_train_statistics(df_train, target_column='target')")
    print("  drift_results = monitor.detect_drift(df_prod, target_column='target')")
    print("  monitor.print_drift_report(drift_results)")
