#!/usr/bin/env python
"""
Quick Start Script for MLOps Project
Automates the complete workflow: data loading, training, and evaluation
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a shell command and log the output"""
    logger.info("=" * 80)
    logger.info(f"Running: {description}")
    logger.info(f"Command: {cmd}")
    logger.info("=" * 80)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {e}")
        logger.error(e.stderr)
        return False


def main():
    """Run the complete MLOps workflow"""
    
    logger.info("\n" + "🚀 " * 40)
    logger.info("STARTING COMPLETE MLOPS WORKFLOW")
    logger.info("🚀 " * 40 + "\n")
    
    # Step 1: Create all dataset versions
    logger.info("\n📊 STEP 1: Creating Dataset Versions")
    logger.info("-" * 80)
    
    for version in [1, 2, 3]:
        if not run_command(
            f"python src/data_loader.py --version {version}",
            f"Creating Dataset Version {version}"
        ):
            logger.error(f"Failed to create dataset v{version}")
            return
    
    # Step 2: Train models with each dataset version
    logger.info("\n🤖 STEP 2: Training Models")
    logger.info("-" * 80)
    
    training_configs = [
        ("v1", "random_forest"),
        ("v1", "xgboost"),
        ("v2", "xgboost"),
        ("v3", "xgboost"),
    ]
    
    for data_version, model_type in training_configs:
        dataset_file = {
            "v1": "v1_california_housing.csv",
            "v2": "v2_filtered_housing.csv",
            "v3": "v3_engineered_housing.csv"
        }[data_version]
        
        if not run_command(
            f"python src/train.py --data_path data/{dataset_file} --model {model_type} --data_version {data_version}",
            f"Training {model_type} with dataset {data_version}"
        ):
            logger.warning(f"Training failed for {model_type} on {data_version}")
    
    # Step 3: Hyperparameter optimization
    logger.info("\n🎯 STEP 3: Hyperparameter Optimization (Advanced Feature)")
    logger.info("-" * 80)
    
    if not run_command(
        "python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv --model xgboost --n_trials 50 --data_version v3",
        "Optimizing XGBoost hyperparameters with Optuna"
    ):
        logger.warning("Hyperparameter optimization failed")
    
    # Step 4: Evaluate and compare all models
    logger.info("\n📊 STEP 4: Model Evaluation and Comparison")
    logger.info("-" * 80)
    
    if not run_command(
        "python src/evaluate.py --compare_all",
        "Comparing all models and generating reports"
    ):
        logger.warning("Evaluation failed")
    
    # Final summary
    logger.info("\n" + "✅ " * 40)
    logger.info("WORKFLOW COMPLETED SUCCESSFULLY!")
    logger.info("✅ " * 40 + "\n")
    
    logger.info("📁 Generated Artifacts:")
    logger.info("  - data/: 3 dataset versions")
    logger.info("  - models/: Trained models")
    logger.info("  - artifacts/: Prediction plots")
    logger.info("  - results/: Comparison reports and Optuna results")
    logger.info("  - mlruns/: MLflow experiment tracking")
    logger.info("")
    logger.info("🌐 View MLflow UI:")
    logger.info("  mlflow ui")
    logger.info("  Then open: http://localhost:5000")
    logger.info("")
    logger.info("📄 Check Documentation:")
    logger.info("  See DOCUMENTATION.md for detailed information")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
