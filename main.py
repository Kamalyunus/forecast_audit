"""
Main entry point for the forecast adjustment system with linear function approximation.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import time

from forecast_environment import ForecastEnvironment
from linear_agent import LinearAgent
from trainer import ForecastAdjustmentTrainer

def load_data(args, logger):
    """
    Load data from files.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        Tuple of (forecast_data, historical_data)
    """
    # Initialize data containers
    forecast_data = None
    historical_data = None
    
    # Load forecast data
    logger.info(f"Loading forecast data from {args.forecast_file}")
    forecast_data = pd.read_csv(args.forecast_file)
    
    # Standardize column names
    if 'product_id' in forecast_data.columns:
        forecast_data = forecast_data.rename(columns={'product_id': 'sku_id'})
    
    # Load historical data if available
    if args.historical_file and os.path.exists(args.historical_file):
        logger.info(f"Loading historical data from {args.historical_file}")
        historical_data = pd.read_csv(args.historical_file)
        
        # Standardize column names
        if 'product_id' in historical_data.columns:
            historical_data = historical_data.rename(columns={'product_id': 'sku_id'})
        if 'date' in historical_data.columns:
            historical_data = historical_data.rename(columns={'date': 'sale_date'})
    
    return forecast_data, historical_data

def main():
    """Main function to run the forecast adjustment system."""
    parser = argparse.ArgumentParser(description='Forecast Adjustment System')
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'evaluate', 'adjust'], 
                       default='train', help='Operation mode')
    
    # Data options
    parser.add_argument('--forecast-file', required=True, help='Path to forecast data CSV')
    parser.add_argument('--historical-file', help='Path to historical sales data CSV')
    
    # Agent options
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Final exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Exploration decay rate')
    
    # Training options
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for updates')
    parser.add_argument('--max-steps', type=int, default=14, help='Maximum steps per episode')
    parser.add_argument('--save-every', type=int, default=50, help='Save model every N episodes')
    
    # Forecast options
    parser.add_argument('--optimize-for', choices=['mape', 'bias', 'both'], default='both',
                       help='Which metric to optimize for')
    parser.add_argument('--validation-length', type=int, default=30, 
                       help='Number of days to use for validation')
    parser.add_argument('--forecast-horizon', type=int, default=7,
                       help='Number of days in the forecast horizon')
    
    # Model options
    parser.add_argument('--model-path', help='Path to saved model')
    parser.add_argument('--output-dir', default='output', help='Directory for outputs')
    
    # Other options
    parser.add_argument('--random-seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set up logger
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'forecast_adjustment.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("ForecastAdjustment")
    
    # Set random seed
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        logger.info(f"Random seed set to {args.random_seed}")
    
    # Load data
    forecast_data, historical_data = load_data(args, logger)
    
    # Create environment
    logger.info("Creating forecast environment")
    env = ForecastEnvironment(
        forecast_data=forecast_data,
        historical_data=historical_data,
        validation_length=args.validation_length,
        forecast_horizon=args.forecast_horizon,
        optimize_for=args.optimize_for,
        logger=logger
    )
    
    # Display environment stats
    logger.info(f"Environment created with {len(env.skus)} SKUs")
    
    # Run in appropriate mode
    if args.mode == 'train':
        # Get dimensions from environment
        forecast_dim, error_dim, feature_dim = env.get_feature_dims()
        action_size = 7  # [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3] × forecast
        
        # Create agent
        logger.info("Creating linear agent")
        agent = LinearAgent(
            feature_dim=feature_dim,
            action_size=action_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay
        )
        
        # Create trainer
        trainer = ForecastAdjustmentTrainer(
            agent=agent,
            environment=env,
            output_dir=args.output_dir,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            save_every=args.save_every,
            optimize_for=args.optimize_for,
            logger=logger
        )
        
        # Train the agent
        logger.info(f"Starting training for {args.episodes} episodes")
        start_time = time.time()
        metrics = trainer.train()
        training_time = time.time() - start_time
        
        # Log results
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final metrics:")
        logger.info(f"  Average Score (last 100): {np.mean(metrics['scores'][-100:]):.2f}")
        logger.info(f"  Average MAPE Improvement (last 100): {np.mean(metrics['mape_improvements'][-100:]):.4f}")
        logger.info(f"  Average Bias Improvement (last 100): {np.mean(metrics['bias_improvements'][-100:]):.4f}")
        
    elif args.mode == 'evaluate':
        # Check if model path is provided
        if not args.model_path:
            logger.error("Model path must be provided for evaluation mode")
            return
        
        if not os.path.exists(args.model_path):
            logger.error(f"Model file {args.model_path} not found")
            return
        
        # Load agent
        logger.info(f"Loading model from {args.model_path}")
        agent = LinearAgent.load(args.model_path)
        
        # Create trainer for evaluation
        trainer = ForecastAdjustmentTrainer(
            agent=agent,
            environment=env,
            output_dir=args.output_dir,
            optimize_for=args.optimize_for,
            logger=logger
        )
        
        # Evaluate the agent
        logger.info("Starting evaluation")
        eval_metrics = trainer.evaluate(num_episodes=10)
        
        # Log results
        logger.info(f"Evaluation completed")
        logger.info(f"Metrics:")
        logger.info(f"  Average Score: {eval_metrics['avg_score']:.2f}")
        logger.info(f"  Average MAPE Improvement: {eval_metrics['avg_mape_improvement']:.4f}")
        logger.info(f"  Average Bias Improvement: {eval_metrics['avg_bias_improvement']:.4f}")
        
        # Log top performing SKUs
        logger.info("Top performing SKUs (by combined improvement):")
        for i, (sku, metrics) in enumerate(eval_metrics['top_skus'][:10]):
            logger.info(f"  {i+1}. {sku}: MAPE Imp = {metrics['mape_improvement']:.4f}, " 
                     f"Bias Imp = {metrics['bias_improvement']:.4f}, "
                     f"Common Adjustment = {metrics['common_adjustment']:.2f}x")
        
    elif args.mode == 'adjust':
        # Check if model path is provided
        if not args.model_path:
            logger.error("Model path must be provided for adjustment mode")
            return
        
        if not os.path.exists(args.model_path):
            logger.error(f"Model file {args.model_path} not found")
            return
        
        # Load agent
        logger.info(f"Loading model from {args.model_path}")
        agent = LinearAgent.load(args.model_path)
        
        # Create trainer for generating adjustments
        trainer = ForecastAdjustmentTrainer(
            agent=agent,
            environment=env,
            output_dir=args.output_dir,
            optimize_for=args.optimize_for,
            logger=logger
        )
        
        # Generate adjusted forecasts
        logger.info("Generating adjusted forecasts")
        adjustments = trainer.generate_adjusted_forecasts(num_days=args.max_steps)
        
        # Save adjustments
        adjustments_path = os.path.join(args.output_dir, "adjusted_forecasts.csv")
        adjustments.to_csv(adjustments_path, index=False)
        logger.info(f"Adjusted forecasts saved to {adjustments_path}")
        
        # Log summary
        logger.info(f"Adjustment summary:")
        logger.info(f"  Total SKUs: {adjustments['sku_id'].nunique()}")
        logger.info(f"  Total days: {adjustments['day'].nunique()}")
        
        # Calculate average adjustment by factor
        action_counts = adjustments['action_idx'].value_counts().sort_index()
        adjustment_factors = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        
        for action_idx, count in action_counts.items():
            percentage = count / len(adjustments) * 100
            factor = adjustment_factors[action_idx] if action_idx < len(adjustment_factors) else 1.0
            logger.info(f"  Factor {factor:.1f}x: {percentage:.1f}% of adjustments")
        
        # Create summary plots
        plt.figure(figsize=(12, 10))
        
        # Distribution of adjustment factors
        plt.subplot(2, 2, 1)
        factor_labels = [f"{f:.1f}x" for f in adjustment_factors]
        counts = [action_counts.get(i, 0) for i in range(len(adjustment_factors))]
        plt.bar(factor_labels, counts)
        plt.title('Adjustment Factor Distribution')
        plt.ylabel('Count')
        
        # Average adjustment by day
        plt.subplot(2, 2, 2)
        day_factors = adjustments.groupby('day')['adjustment_factor'].mean()
        plt.plot(day_factors.index, day_factors.values, 'o-')
        plt.title('Average Adjustment Factor by Day')
        plt.xlabel('Day')
        plt.ylabel('Avg Factor')
        plt.grid(True, alpha=0.3)
        
        # Original vs Adjusted Forecast scatter
        plt.subplot(2, 2, 3)
        plt.scatter(adjustments['original_forecast'], 
                   adjustments['adjusted_forecast'], 
                   alpha=0.3)
        max_val = max(adjustments['original_forecast'].max(), 
                      adjustments['adjusted_forecast'].max())
        plt.plot([0, max_val], [0, max_val], 'r--')  # Diagonal line
        plt.title('Original vs Adjusted Forecast')
        plt.xlabel('Original Forecast')
        plt.ylabel('Adjusted Forecast')
        plt.grid(True, alpha=0.3)
        
        # Adjustment percentage distribution
        plt.subplot(2, 2, 4)
        # Calculate percentage adjustments
        pct_changes = ((adjustments['adjusted_forecast'] - adjustments['original_forecast']) 
                      / adjustments['original_forecast'].clip(lower=1e-8)) * 100
        # Remove extreme values for better visualization
        pct_changes = pct_changes.clip(lower=-50, upper=50)
        plt.hist(pct_changes, bins=20)
        plt.title('Adjustment Percentage Distribution')
        plt.xlabel('Adjustment (%)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "adjustment_summary.png"))
        logger.info(f"Summary plots saved to {os.path.join(args.output_dir, 'adjustment_summary.png')}")
    
    logger.info("Operation completed successfully")


if __name__ == "__main__":
    main()