"""
Training Module with MLflow Integration
Supports multiple models and comprehensive experiment tracking
"""

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import monitoring module
try:
    from monitoring import DataDriftMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from src.monitoring import DataDriftMonitor
        MONITORING_AVAILABLE = True
    except ImportError:
        MONITORING_AVAILABLE = False
        print("Warning: Monitoring module not available")

# Try to import XGBoost (it's optional)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Unified trainer for multiple regression models"""
    
    MODELS = {
        'random_forest': {
            'class': RandomForestRegressor,
            'default_params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        'gradient_boosting': {
            'class': GradientBoostingRegressor,
            'default_params': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'min_samples_split': 5,
                'random_state': 42
            }
        },
        'ridge': {
            'class': Ridge,
            'default_params': {
                'alpha': 1.0,
                'random_state': 42
            }
        }
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        MODELS['xgboost'] = {
            'class': XGBRegressor,
            'default_params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
        }
    
    def __init__(self, model_name='random_forest', custom_params=None):
        """
        Initialize trainer
        
        Args:
            model_name (str): Name of model to use
            custom_params (dict): Custom hyperparameters
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(self.MODELS.keys())}")
        
        self.model_name = model_name
        self.model_config = self.MODELS[model_name]
        
        # Merge default and custom params
        self.params = self.model_config['default_params'].copy()
        if custom_params:
            self.params.update(custom_params)
        
        # Initialize model
        self.model = self.model_config['class'](**self.params)
        
        logger.info(f"Initialized {model_name} with params: {self.params}")
    
    def train(self, X_train, y_train):
        """Train the model"""
        logger.info(f"Training {self.model_name}...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return training_time
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model and return metrics
        
        Returns:
            dict: Dictionary of metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }
        
        logger.info(f"Evaluation Metrics:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def save_model(self, filepath):
        """Save trained model"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def plot_predictions(self, y_test, predictions, save_path=None):
        """Create prediction vs actual plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        ax1.scatter(y_test, predictions, alpha=0.5)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'{self.model_name.upper()} - Predictions vs Actual')
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_test - predictions
        ax2.scatter(predictions, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig


def load_data(data_path, test_size=0.2, random_state=42):
    """Load and split data"""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Data loaded: {len(X_train)} train, {len(X_test)} test samples")
    logger.info(f"Features: {list(X.columns)}")
    
    return X_train, X_test, y_train, y_test


def train_with_mlflow(data_path, model_name='random_forest', experiment_name='california-housing',
                      custom_params=None, data_version='v1', enable_monitoring=True,
                      target_column='MedHouseVal', drift_threshold=2.0):
    """
    Complete training pipeline with MLflow tracking and monitoring integration
    
    Args:
        data_path (str): Path to dataset
        model_name (str): Model to use
        experiment_name (str): MLflow experiment name
        custom_params (dict): Custom hyperparameters
        data_version (str): Version of dataset being used
        enable_monitoring (bool): Generate monitoring baseline statistics
        target_column (str): Name of target column for monitoring
        drift_threshold (float): Threshold multiplier for drift detection
    
    Returns:
        tuple: (model, metrics, run_id)
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        # Log dataset info
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("model_type", model_name)
        
        # Load data
        X_train, X_test, y_train, y_test = load_data(data_path)
        
        # Log data characteristics
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        
        # Initialize and train model
        trainer = ModelTrainer(model_name, custom_params)
        
        # Log hyperparameters
        for param_name, param_value in trainer.params.items():
            mlflow.log_param(param_name, param_value)
        
        # Train
        training_time = trainer.train(X_train, y_train)
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)
        
        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Create and log prediction plot
        predictions = trainer.predict(X_test)
        plot_path = f"artifacts/predictions_{model_name}_{data_version}.png"
        trainer.plot_predictions(y_test, predictions, save_path=plot_path)
        mlflow.log_artifact(plot_path)
        
        # Save and log model
        model_path = f"models/{model_name}_{data_version}.pkl"
        trainer.save_model(model_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(trainer.model, "model")
        
        # Log model file as artifact
        mlflow.log_artifact(model_path)
        
        # ===== MONITORING INTEGRATION =====
        if enable_monitoring and MONITORING_AVAILABLE:
            logger.info("\n" + "=" * 60)
            logger.info("GENERATING MONITORING BASELINE STATISTICS")
            logger.info("=" * 60)
            
            try:
                # Load full dataset for monitoring baseline
                df_full = pd.read_csv(data_path)
                
                # Setup monitoring paths
                monitoring_dir = Path("artifacts/monitoring")
                monitoring_dir.mkdir(parents=True, exist_ok=True)
                
                stats_filename = f"train_stats_{data_version}_{model_name}.json"
                stats_path = monitoring_dir / stats_filename
                
                # Initialize monitor
                monitor = DataDriftMonitor(
                    stats_path=str(stats_path),
                    threshold_multiplier=drift_threshold
                )
                
                # Calculate and save statistics
                logger.info(f"Calculating baseline statistics for monitoring...")
                monitor.save_train_statistics(df_full, target_column=target_column)
                
                # Log monitoring config to MLflow
                mlflow.log_param("monitoring_enabled", True)
                mlflow.log_param("monitoring_target_column", target_column)
                mlflow.log_param("monitoring_drift_threshold", drift_threshold)
                mlflow.log_metric("monitoring_baseline_samples", len(df_full))
                mlflow.log_metric("monitoring_features_tracked", len(df_full.columns) - 1)
                
                # Log monitoring stats as artifact
                mlflow.log_artifact(str(stats_path), artifact_path="monitoring")
                
                logger.info(f"✓ Monitoring baseline saved: {stats_path}")
                logger.info(f"✓ Artifacts logged to MLflow run: {run_id}")
                
            except Exception as e:
                logger.warning(f"Failed to generate monitoring baseline: {e}")
                mlflow.log_param("monitoring_enabled", False)
                mlflow.log_param("monitoring_error", str(e))
        else:
            mlflow.log_param("monitoring_enabled", False)
            if enable_monitoring and not MONITORING_AVAILABLE:
                logger.warning("Monitoring requested but module not available")
        
        logger.info("=" * 60)
        logger.info(f"Training completed successfully!")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Data Version: {data_version}")
        logger.info("=" * 60)
        
        return trainer.model, metrics, run_id


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Train California Housing price prediction model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data CSV')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'ridge', 'xgboost'],
                        help='Model type to train')
    parser.add_argument('--data_version', type=str, default='v1',
                        help='Version of dataset (v1, v2, v3)')
    parser.add_argument('--experiment', type=str, default='california-housing',
                        help='MLflow experiment name')
    parser.add_argument('--n_estimators', type=int, default=None,
                        help='Number of estimators (for tree-based models)')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='Max depth (for tree-based models)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (for boosting models)')
    parser.add_argument('--enable-monitoring', action='store_true', default=True,
                        help='Generate monitoring baseline statistics (default: True)')
    parser.add_argument('--no-monitoring', dest='enable_monitoring', action='store_false',
                        help='Disable monitoring baseline generation')
    parser.add_argument('--drift-threshold', type=float, default=2.0,
                        help='Drift detection threshold multiplier (default: 2.0)')
    
    args = parser.parse_args()
    
    # Build custom params from arguments
    custom_params = {}
    if args.n_estimators is not None:
        custom_params['n_estimators'] = args.n_estimators
    if args.max_depth is not None:
        custom_params['max_depth'] = args.max_depth
    if args.learning_rate is not None:
        custom_params['learning_rate'] = args.learning_rate
    
    # Train model
    model, metrics, run_id = train_with_mlflow(
        data_path=args.data_path,
        model_name=args.model,
        experiment_name=args.experiment,
        custom_params=custom_params if custom_params else None,
        data_version=args.data_version,
        enable_monitoring=args.enable_monitoring,
        drift_threshold=args.drift_threshold
    )
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data Version: {args.data_version}")
    print(f"MLflow Run ID: {run_id}")
    print(f"\nMetrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name.upper()}: {metric_value:.4f}")
    print("=" * 60)
    print("\nView results in MLflow UI:")
    print("  mlflow ui")
    print("  Then open: http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
