"""
Model Evaluation and Comparison Module
Compare different models and dataset versions
"""

import argparse
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare multiple ML models and experiments"""
    
    def __init__(self, experiment_names=None):
        """
        Initialize comparator
        
        Args:
            experiment_names (list): List of MLflow experiment names to compare
        """
        self.experiment_names = experiment_names or ['california-housing', 'optuna-tuning']
        self.results = []
        
    def fetch_mlflow_runs(self):
        """Fetch all runs from specified experiments"""
        logger.info("Fetching runs from MLflow...")
        
        client = mlflow.tracking.MlflowClient()
        
        for exp_name in self.experiment_names:
            try:
                experiment = client.get_experiment_by_name(exp_name)
                if experiment is None:
                    logger.warning(f"Experiment '{exp_name}' not found")
                    continue
                
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"]
                )
                
                logger.info(f"Found {len(runs)} runs in experiment '{exp_name}'")
                
                for run in runs:
                    run_data = {
                        'run_id': run.info.run_id,
                        'experiment': exp_name,
                        'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
                        'status': run.info.status,
                    }
                    
                    # Add parameters
                    run_data.update({f'param_{k}': v for k, v in run.data.params.items()})
                    
                    # Add metrics
                    run_data.update({f'metric_{k}': float(v) for k, v in run.data.metrics.items()})
                    
                    self.results.append(run_data)
                    
            except Exception as e:
                logger.error(f"Error fetching experiment '{exp_name}': {e}")
        
        logger.info(f"Total runs fetched: {len(self.results)}")
        return self.results
    
    def create_comparison_dataframe(self):
        """Create a DataFrame for easy comparison"""
        if not self.results:
            self.fetch_mlflow_runs()
        
        df = pd.DataFrame(self.results)
        
        # Select key columns
        key_columns = ['run_id', 'experiment', 'start_time', 'status']
        
        # Add important params and metrics
        param_cols = [c for c in df.columns if c.startswith('param_')]
        metric_cols = [c for c in df.columns if c.startswith('metric_')]
        
        df = df[key_columns + param_cols + metric_cols]
        
        return df
    
    def get_best_runs(self, metric='metric_rmse', n=5):
        """
        Get best performing runs
        
        Args:
            metric (str): Metric to sort by
            n (int): Number of top runs to return
            
        Returns:
            pd.DataFrame: Top N runs
        """
        df = self.create_comparison_dataframe()
        
        if metric not in df.columns:
            logger.error(f"Metric '{metric}' not found in runs")
            return None
        
        # Sort by metric (lower is better for RMSE)
        df_sorted = df.sort_values(metric).head(n)
        
        logger.info(f"\nTop {n} runs by {metric}:")
        logger.info("=" * 80)
        for idx, row in df_sorted.iterrows():
            logger.info(f"Run ID: {row['run_id'][:8]}... | "
                       f"Model: {row.get('param_model_type', 'N/A')} | "
                       f"Data: {row.get('param_data_version', 'N/A')} | "
                       f"{metric}: {row[metric]:.4f}")
        
        return df_sorted
    
    def compare_by_data_version(self, output_path='results/comparison_by_version.png'):
        """Compare model performance across different data versions"""
        df = self.create_comparison_dataframe()
        
        if 'param_data_version' not in df.columns or 'metric_rmse' not in df.columns:
            logger.warning("Required columns not found for comparison")
            return None
        
        # Group by data version
        version_stats = df.groupby('param_data_version').agg({
            'metric_rmse': ['mean', 'min', 'max', 'std'],
            'metric_r2': ['mean', 'min', 'max'],
            'metric_mae': ['mean', 'min', 'max']
        }).round(4)
        
        logger.info("\n" + "=" * 80)
        logger.info("Performance by Data Version:")
        logger.info("=" * 80)
        print(version_stats)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = ['metric_rmse', 'metric_mae', 'metric_r2']
        titles = ['RMSE by Data Version', 'MAE by Data Version', 'R² by Data Version']
        
        for ax, metric, title in zip(axes, metrics, titles):
            if metric in df.columns:
                df.boxplot(column=metric, by='param_data_version', ax=ax)
                ax.set_title(title)
                ax.set_xlabel('Data Version')
                ax.set_ylabel(metric.replace('metric_', '').upper())
                plt.sca(ax)
                plt.xticks(rotation=0)
        
        plt.suptitle('Model Performance Comparison Across Data Versions', y=1.02)
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {output_path}")
        
        return fig, version_stats
    
    def compare_by_model_type(self, output_path='results/comparison_by_model.png'):
        """Compare different model types"""
        df = self.create_comparison_dataframe()
        
        if 'param_model_type' not in df.columns:
            logger.warning("Model type information not found")
            return None
        
        # Group by model type
        model_stats = df.groupby('param_model_type').agg({
            'metric_rmse': ['mean', 'min', 'max', 'std'],
            'metric_r2': ['mean', 'min', 'max'],
            'metric_mae': ['mean', 'min', 'max']
        }).round(4)
        
        logger.info("\n" + "=" * 80)
        logger.info("Performance by Model Type:")
        logger.info("=" * 80)
        print(model_stats)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # RMSE comparison
        if 'metric_rmse' in df.columns:
            df.boxplot(column='metric_rmse', by='param_model_type', ax=axes[0])
            axes[0].set_title('RMSE by Model Type')
            axes[0].set_xlabel('Model Type')
            axes[0].set_ylabel('RMSE')
        
        # R² comparison
        if 'metric_r2' in df.columns:
            df.boxplot(column='metric_r2', by='param_model_type', ax=axes[1])
            axes[1].set_title('R² Score by Model Type')
            axes[1].set_xlabel('Model Type')
            axes[1].set_ylabel('R²')
        
        plt.suptitle('Model Type Comparison', y=1.02)
        plt.tight_layout()
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {output_path}")
        
        return fig, model_stats
    
    def generate_report(self, output_path='results/comparison_report.txt'):
        """Generate comprehensive comparison report"""
        df = self.create_comparison_dataframe()
        
        report = []
        report.append("=" * 80)
        report.append("ML EXPERIMENTS COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Runs: {len(df)}")
        report.append("")
        
        # Overall statistics
        report.append("-" * 80)
        report.append("OVERALL STATISTICS")
        report.append("-" * 80)
        
        for metric in ['metric_rmse', 'metric_mae', 'metric_r2']:
            if metric in df.columns:
                report.append(f"\n{metric.replace('metric_', '').upper()}:")
                report.append(f"  Mean: {df[metric].mean():.4f}")
                report.append(f"  Std: {df[metric].std():.4f}")
                report.append(f"  Min: {df[metric].min():.4f}")
                report.append(f"  Max: {df[metric].max():.4f}")
        
        # Best runs
        report.append("\n" + "-" * 80)
        report.append("TOP 5 BEST RUNS (by RMSE)")
        report.append("-" * 80)
        
        if 'metric_rmse' in df.columns:
            best_runs = df.nsmallest(5, 'metric_rmse')
            for idx, (_, row) in enumerate(best_runs.iterrows(), 1):
                report.append(f"\n#{idx}")
                report.append(f"  Run ID: {row['run_id']}")
                report.append(f"  Model: {row.get('param_model_type', 'N/A')}")
                report.append(f"  Data Version: {row.get('param_data_version', 'N/A')}")
                report.append(f"  RMSE: {row.get('metric_rmse', 'N/A'):.4f}")
                report.append(f"  R²: {row.get('metric_r2', 'N/A'):.4f}")
                report.append(f"  MAE: {row.get('metric_mae', 'N/A'):.4f}")
        
        # Data version comparison
        if 'param_data_version' in df.columns:
            report.append("\n" + "-" * 80)
            report.append("PERFORMANCE BY DATA VERSION")
            report.append("-" * 80)
            
            for version in df['param_data_version'].unique():
                if pd.isna(version):
                    continue
                version_df = df[df['param_data_version'] == version]
                report.append(f"\n{version}:")
                report.append(f"  Runs: {len(version_df)}")
                if 'metric_rmse' in df.columns:
                    report.append(f"  Avg RMSE: {version_df['metric_rmse'].mean():.4f}")
                    report.append(f"  Best RMSE: {version_df['metric_rmse'].min():.4f}")
                if 'metric_r2' in df.columns:
                    report.append(f"  Avg R²: {version_df['metric_r2'].mean():.4f}")
        
        # Model type comparison
        if 'param_model_type' in df.columns:
            report.append("\n" + "-" * 80)
            report.append("PERFORMANCE BY MODEL TYPE")
            report.append("-" * 80)
            
            for model in df['param_model_type'].unique():
                if pd.isna(model):
                    continue
                model_df = df[df['param_model_type'] == model]
                report.append(f"\n{model}:")
                report.append(f"  Runs: {len(model_df)}")
                if 'metric_rmse' in df.columns:
                    report.append(f"  Avg RMSE: {model_df['metric_rmse'].mean():.4f}")
                    report.append(f"  Best RMSE: {model_df['metric_rmse'].min():.4f}")
                if 'metric_r2' in df.columns:
                    report.append(f"  Avg R²: {model_df['metric_r2'].mean():.4f}")
        
        report.append("\n" + "=" * 80)
        
        # Write to file
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write('\n'.join(report))
            logger.info(f"Report saved to {output_path}")
        
        # Print to console
        print('\n'.join(report))
        
        return '\n'.join(report)


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Compare ML experiments')
    parser.add_argument('--experiments', nargs='+', 
                        default=['california-housing', 'optuna-tuning'],
                        help='MLflow experiment names to compare')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--compare_all', action='store_true',
                        help='Generate all comparison plots and reports')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ModelComparator(experiment_names=args.experiments)
    
    # Fetch runs
    comparator.fetch_mlflow_runs()
    
    if args.compare_all:
        # Generate all comparisons
        logger.info("Generating comprehensive comparison...")
        
        # Best runs
        comparator.get_best_runs(n=5)
        
        # Compare by data version
        comparator.compare_by_data_version(f'{args.output_dir}/comparison_by_version.png')
        
        # Compare by model type
        comparator.compare_by_model_type(f'{args.output_dir}/comparison_by_model.png')
        
        # Generate report
        comparator.generate_report(f'{args.output_dir}/comparison_report.txt')
        
        logger.info("\n" + "=" * 80)
        logger.info("Comparison complete! Check the results directory.")
        logger.info("=" * 80)
    else:
        # Just show best runs
        comparator.get_best_runs(n=10)


if __name__ == "__main__":
    main()
