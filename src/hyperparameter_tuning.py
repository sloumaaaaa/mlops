"""
Hyperparameter Tuning Module with Optuna
Advanced feature for automated hyperparameter optimization
"""

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from pathlib import Path
import joblib
import json

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, model_type='xgboost', n_trials=100, cv_folds=5):
        """
        Initialize tuner
        
        Args:
            model_type (str): Type of model to tune
            n_trials (int): Number of optimization trials
            cv_folds (int): Number of cross-validation folds
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = None
        self.best_score = None
        self.study = None
        
    def _objective_random_forest(self, trial, X_train, y_train):
        """Objective function for Random Forest"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        
        # Cross-validation
        scores = cross_val_score(
            model, X_train, y_train,
            cv=self.cv_folds,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        return -scores.mean()  # Return positive RMSE
    
    def _objective_gradient_boosting(self, trial, X_train, y_train):
        """Objective function for Gradient Boosting"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42
        }
        
        model = GradientBoostingRegressor(**params)
        
        scores = cross_val_score(
            model, X_train, y_train,
            cv=self.cv_folds,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        return -scores.mean()
    
    def _objective_xgboost(self, trial, X_train, y_train):
        """Objective function for XGBoost"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = XGBRegressor(**params)
        
        scores = cross_val_score(
            model, X_train, y_train,
            cv=self.cv_folds,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        return -scores.mean()
    
    def optimize(self, X_train, y_train, experiment_name='optuna-tuning'):
        """
        Run hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training target
            experiment_name: MLflow experiment name
            
        Returns:
            dict: Best parameters found
        """
        logger.info("=" * 60)
        logger.info(f"Starting Optuna hyperparameter optimization for {self.model_type}")
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info(f"CV folds: {self.cv_folds}")
        logger.info("=" * 60)
        
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)
        
        # Select objective function
        objective_functions = {
            'random_forest': self._objective_random_forest,
            'gradient_boosting': self._objective_gradient_boosting,
            'xgboost': self._objective_xgboost
        }
        
        if self.model_type not in objective_functions:
            raise ValueError(f"Model type {self.model_type} not supported")
        
        objective_fn = objective_functions[self.model_type]
        
        # Create study
        self.study = optuna.create_study(
            direction='minimize',
            study_name=f'{self.model_type}_optimization'
        )
        
        # MLflow callback
        mlflc = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name='cv_rmse'
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: objective_fn(trial, X_train, y_train),
            n_trials=self.n_trials,
            callbacks=[mlflc],
            show_progress_bar=True
        )
        
        # Get best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info("=" * 60)
        logger.info("Optimization completed!")
        logger.info(f"Best RMSE: {self.best_score:.4f}")
        logger.info(f"Best parameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info("=" * 60)
        
        return self.best_params
    
    def train_best_model(self, X_train, y_train, X_test, y_test):
        """
        Train final model with best parameters
        
        Returns:
            tuple: (model, metrics)
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
        
        logger.info("Training final model with best parameters...")
        
        # Create model with best params
        if self.model_type == 'random_forest':
            model = RandomForestRegressor(**self.best_params)
        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(**self.best_params)
        elif self.model_type == 'xgboost':
            model = XGBRegressor(**self.best_params)
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        logger.info("Final Model Metrics:")
        logger.info(f"  Train RMSE: {metrics['train_rmse']:.4f}")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"  Train R²: {metrics['train_r2']:.4f}")
        logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
        
        return model, metrics
    
    def save_results(self, output_dir='results/optuna'):
        """Save optimization results"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save best params
        params_path = f'{output_dir}/{self.model_type}_best_params.json'
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        logger.info(f"Best parameters saved to {params_path}")
        
        # Save study
        study_path = f'{output_dir}/{self.model_type}_study.pkl'
        joblib.dump(self.study, study_path)
        logger.info(f"Study saved to {study_path}")
        
        return params_path, study_path
    
    def plot_optimization_history(self, save_path=None):
        """Plot optimization history"""
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances
            import matplotlib.pyplot as plt
            
            fig1 = plot_optimization_history(self.study)
            fig2 = plot_param_importances(self.study)
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig1.write_image(f"{save_path}_history.png")
                fig2.write_image(f"{save_path}_importance.png")
                logger.info(f"Plots saved to {save_path}")
            
            return fig1, fig2
        except ImportError:
            logger.warning("Plotly not available for visualization. Install with: pip install plotly kaleido")
            return None, None


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data CSV')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['random_forest', 'gradient_boosting', 'xgboost'],
                        help='Model type to tune')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--data_version', type=str, default='v1',
                        help='Version of dataset')
    parser.add_argument('--output_dir', type=str, default='results/optuna',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Data loaded: {len(X_train)} train, {len(X_test)} test samples")
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        model_type=args.model,
        n_trials=args.n_trials,
        cv_folds=args.cv_folds
    )
    
    # Optimize
    best_params = tuner.optimize(X_train, y_train)
    
    # Train final model
    model, metrics = tuner.train_best_model(X_train, y_train, X_test, y_test)
    
    # Save results
    tuner.save_results(args.output_dir)
    
    # Save final model
    model_path = f'models/{args.model}_optimized_{args.data_version}.pkl'
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Final model saved to {model_path}")
    
    # Try to plot (if plotly available)
    tuner.plot_optimization_history(f'{args.output_dir}/{args.model}_plots')
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION COMPLETED")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Trials: {args.n_trials}")
    print(f"Best CV RMSE: {tuner.best_score:.4f}")
    print(f"\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\nFinal Test Metrics:")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
