"""
Preprocessing Module
Handles data preprocessing and feature scaling
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocessor for California Housing data"""
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize preprocessor
        
        Args:
            scaler_type (str): Type of scaler ('standard' or 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else RobustScaler()
        self.feature_names = None
        
    def fit_transform(self, df):
        """
        Fit and transform the data
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        # Separate features and target
        if 'MedHouseVal' in df.columns:
            X = df.drop('MedHouseVal', axis=1)
            y = df['MedHouseVal']
        else:
            raise ValueError("Target column 'MedHouseVal' not found in dataframe")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Fit and transform
        X_scaled = self.scaler.fit_transform(X)
        
        # Create DataFrame with scaled features
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Add target back
        X_scaled_df['MedHouseVal'] = y.values
        
        logger.info(f"Data scaled using {self.scaler_type} scaler")
        logger.info(f"Features: {self.feature_names}")
        
        return X_scaled_df
    
    def transform(self, df):
        """
        Transform new data using fitted scaler
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        if 'MedHouseVal' in df.columns:
            X = df.drop('MedHouseVal', axis=1)
            y = df['MedHouseVal']
        else:
            X = df
            y = None
        
        # Transform
        X_scaled = self.scaler.transform(X)
        
        # Create DataFrame
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        if y is not None:
            X_scaled_df['MedHouseVal'] = y.values
        
        return X_scaled_df
    
    def save(self, filepath):
        """Save the preprocessor"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load a saved preprocessor"""
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(input_path, output_dir='data/processed', scaler_type='standard'):
    """
    Complete preprocessing pipeline
    
    Args:
        input_path (str): Path to input CSV
        output_dir (str): Output directory for processed data
        scaler_type (str): Type of scaler to use
        
    Returns:
        tuple: Paths to saved files
    """
    logger.info("=" * 60)
    logger.info(f"Preprocessing pipeline for: {input_path}")
    logger.info("=" * 60)
    
    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded data: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(scaler_type=scaler_type)
    
    # Fit and transform
    df_scaled = preprocessor.fit_transform(df)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    input_filename = Path(input_path).stem
    output_path = f"{output_dir}/{input_filename}_processed.csv"
    scaler_path = f"{output_dir}/{input_filename}_scaler.pkl"
    
    # Save processed data
    df_scaled.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    
    # Save scaler
    preprocessor.save(scaler_path)
    
    # Print statistics
    logger.info("=" * 60)
    logger.info("Processed Data Statistics:")
    logger.info("=" * 60)
    print(df_scaled.describe())
    
    return output_path, scaler_path


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Preprocess California Housing data')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--scaler', type=str, default='standard', choices=['standard', 'robust'],
                        help='Type of scaler to use')
    
    args = parser.parse_args()
    
    preprocess_pipeline(args.input, args.output_dir, args.scaler)


if __name__ == "__main__":
    main()
