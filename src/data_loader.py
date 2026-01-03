"""
Data Loader Module for California Housing Dataset
Handles different versions of the dataset with various transformations
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_raw_data():
    """Load the raw California Housing dataset from sklearn"""
    logger.info("Loading California Housing dataset...")
    try:
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
    except Exception as e:
        logger.warning(f"Could not download dataset: {e}")
        logger.info("Using local data or generating synthetic data...")
        # Generate synthetic data that mimics California Housing
        np.random.seed(42)
        n_samples = 20640
        
        data = {
            'MedInc': np.random.uniform(0.5, 15, n_samples),
            'HouseAge': np.random.uniform(1, 52, n_samples),
            'AveRooms': np.random.uniform(1, 10, n_samples),
            'AveBedrms': np.random.uniform(0.5, 5, n_samples),
            'Population': np.random.uniform(100, 3500, n_samples),
            'AveOccup': np.random.uniform(1, 6, n_samples),
            'Latitude': np.random.uniform(32.5, 42, n_samples),
            'Longitude': np.random.uniform(-124, -114, n_samples),
        }
        
        # Generate target with some correlation to features
        data['MedHouseVal'] = (
            data['MedInc'] * 0.4 + 
            (52 - data['HouseAge']) * 0.02 +
            data['AveRooms'] * 0.1 +
            np.random.normal(0, 0.5, n_samples)
        ).clip(0.15, 5.0)
        
        df = pd.DataFrame(data)
        logger.info("Synthetic California Housing-style dataset created")
    
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_version_1(output_path='data/v1_california_housing.csv'):
    """
    Version 1: Original dataset without modifications
    """
    logger.info("Creating Version 1: Original dataset")
    df = load_raw_data()
    
    # Create data directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Version 1 saved to {output_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def create_version_2(output_path='data/v2_filtered_housing.csv'):
    """
    Version 2: Filtered dataset
    - Remove outliers in target variable (MedHouseVal > 5.0)
    - Focus on coastal regions (Longitude < -119)
    - Remove missing values if any
    """
    logger.info("Creating Version 2: Filtered dataset")
    df = load_raw_data()
    
    initial_rows = len(df)
    
    # Filter outliers in target variable
    df = df[df['MedHouseVal'] <= 5.0]
    logger.info(f"Removed {initial_rows - len(df)} outliers (MedHouseVal > 5.0)")
    
    # Focus on coastal regions for better predictions
    df = df[df['Longitude'] < -119.0]
    logger.info(f"Filtered to coastal regions: {len(df)} rows remaining")
    
    # Remove any missing values
    df = df.dropna()
    logger.info(f"After removing NaN: {len(df)} rows")
    
    # Create data directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Version 2 saved to {output_path}")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Data reduction: {((initial_rows - len(df)) / initial_rows * 100):.2f}%")
    
    return df


def create_version_3(output_path='data/v3_engineered_housing.csv'):
    """
    Version 3: Feature engineered dataset
    - All Version 2 filters
    - New features:
      * rooms_per_household
      * bedrooms_ratio
      * population_density
      * income_category
    """
    logger.info("Creating Version 3: Feature engineered dataset")
    
    # Start with Version 2 data
    df = load_raw_data()
    
    # Apply V2 filters
    df = df[df['MedHouseVal'] <= 5.0]
    df = df[df['Longitude'] < -119.0]
    df = df.dropna()
    
    # Feature Engineering
    logger.info("Adding engineered features...")
    
    # 1. Rooms per household
    df['rooms_per_household'] = df['AveRooms'] * df['AveOccup']
    
    # 2. Bedrooms ratio (how many bedrooms relative to total rooms)
    df['bedrooms_ratio'] = df['AveBedrms'] / df['AveRooms']
    
    # 3. Population density (people per household)
    df['population_density'] = df['Population'] / df['AveOccup']
    
    # 4. Income category (Low, Medium, High, Very High)
    df['income_category'] = pd.cut(
        df['MedInc'],
        bins=[0, 2.5, 4.5, 6.5, np.inf],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Convert categorical to numeric for ML
    df['income_category_encoded'] = df['income_category'].cat.codes
    
    # Drop the categorical column (keep encoded version)
    df = df.drop('income_category', axis=1)
    
    logger.info(f"New features added: rooms_per_household, bedrooms_ratio, population_density, income_category_encoded")
    
    # Create data directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Version 3 saved to {output_path}")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Load and prepare California Housing dataset')
    parser.add_argument('--version', type=int, default=1, choices=[1, 2, 3],
                        help='Dataset version to create (1, 2, or 3)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for the dataset')
    
    args = parser.parse_args()
    
    # Map version to function and filename
    version_functions = {
        1: (create_version_1, f'{args.output_dir}/v1_california_housing.csv'),
        2: (create_version_2, f'{args.output_dir}/v2_filtered_housing.csv'),
        3: (create_version_3, f'{args.output_dir}/v3_engineered_housing.csv')
    }
    
    func, output_path = version_functions[args.version]
    df = func(output_path)
    
    logger.info("=" * 60)
    logger.info("Dataset Statistics:")
    logger.info("=" * 60)
    print(df.describe())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
