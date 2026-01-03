#!/usr/bin/env python
"""
Dataset Change Manager
Script to demonstrate dataset versioning with DVC
"""

import subprocess
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_git_command(cmd):
    """Execute git command"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.stderr and 'warning' not in result.stderr.lower():
        logger.error(result.stderr)
    return result.returncode == 0


def run_dvc_command(cmd):
    """Execute DVC command"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    return result.returncode == 0


def change_dataset(version, commit_message=None):
    """
    Change dataset version and track with DVC and Git
    
    Args:
        version (int): Dataset version (1, 2, or 3)
        commit_message (str): Custom commit message
    """
    logger.info("=" * 80)
    logger.info(f"Changing to Dataset Version {version}")
    logger.info("=" * 80)
    
    # Create dataset
    logger.info(f"Creating dataset v{version}...")
    result = subprocess.run(
        f"python src/data_loader.py --version {version}",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Failed to create dataset v{version}")
        logger.error(result.stderr)
        return False
    
    logger.info(result.stdout)
    
    # Map version to filename
    dataset_files = {
        1: "data/v1_california_housing.csv",
        2: "data/v2_filtered_housing.csv",
        3: "data/v3_engineered_housing.csv"
    }
    
    dataset_file = dataset_files[version]
    
    # Check if file exists
    if not Path(dataset_file).exists():
        logger.error(f"Dataset file not found: {dataset_file}")
        return False
    
    # Track with DVC
    logger.info(f"Tracking {dataset_file} with DVC...")
    if not run_dvc_command(f"dvc add {dataset_file}"):
        logger.error("Failed to add dataset to DVC")
        return False
    
    # Commit to Git
    logger.info("Committing to Git...")
    
    # Add DVC files
    run_git_command(f"git add {dataset_file}.dvc .gitignore")
    
    # Create commit message
    if commit_message is None:
        commit_messages = {
            1: "Add dataset V1 - Original California Housing data",
            2: "Add dataset V2 - Filtered data (removed outliers, coastal regions)",
            3: "Add dataset V3 - Feature engineered data (new features added)"
        }
        commit_message = commit_messages[version]
    
    if not run_git_command(f'git commit -m "{commit_message}"'):
        logger.warning("Git commit failed (maybe already committed or no changes)")
    
    logger.info("=" * 80)
    logger.info(f"✅ Dataset v{version} successfully versioned with DVC and Git!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info(f"1. Train model: python src/train.py --data_path {dataset_file} --model xgboost --data_version v{version}")
    logger.info("2. Push to DVC remote: dvc push (if remote configured)")
    logger.info("3. Push to Git: git push origin main")
    logger.info("=" * 80)
    
    return True


def demonstrate_all_versions():
    """Demonstrate creating all 3 dataset versions"""
    logger.info("\n" + "🔄 " * 40)
    logger.info("DEMONSTRATING DATASET VERSIONING - ALL 3 VERSIONS")
    logger.info("🔄 " * 40 + "\n")
    
    for version in [1, 2, 3]:
        if not change_dataset(version):
            logger.error(f"Failed at version {version}")
            return False
        
        logger.info("\n" + "-" * 80 + "\n")
    
    logger.info("\n" + "✅ " * 40)
    logger.info("ALL DATASET VERSIONS CREATED AND VERSIONED!")
    logger.info("✅ " * 40 + "\n")
    
    # Show Git history
    logger.info("Git commit history:")
    run_git_command("git log --oneline -5")
    
    # Show DVC status
    logger.info("\nDVC status:")
    run_dvc_command("dvc status")
    
    return True


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Dataset versioning manager for MLOps project')
    parser.add_argument('--version', type=int, choices=[1, 2, 3],
                        help='Dataset version to create and version')
    parser.add_argument('--all', action='store_true',
                        help='Create and version all dataset versions')
    parser.add_argument('--message', type=str,
                        help='Custom commit message')
    
    args = parser.parse_args()
    
    # Initialize DVC if needed
    if not Path('.dvc').exists():
        logger.info("Initializing DVC...")
        run_dvc_command("dvc init")
        run_git_command("git add .dvc .dvcignore")
        run_git_command('git commit -m "Initialize DVC"')
    
    # Initialize Git if needed
    if not Path('.git').exists():
        logger.info("Initializing Git repository...")
        run_git_command("git init")
        run_git_command("git add .")
        run_git_command('git commit -m "Initial commit"')
    
    if args.all:
        demonstrate_all_versions()
    elif args.version:
        change_dataset(args.version, args.message)
    else:
        logger.info("Please specify --version or --all")
        logger.info("Examples:")
        logger.info("  python change_dataset.py --version 1")
        logger.info("  python change_dataset.py --all")


if __name__ == "__main__":
    main()
