"""
Main entry point for the inventory management system with linear function approximation.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time

from inventory_rl.environment.inventory_env import InventoryEnvironment
from inventory_rl.utils.config import Config, RewardConfig
from inventory_rl.utils.logger import setup_logger
from linear_agent import LinearAgent, ClusteredLinearAgent
from trainer import Trainer
from clustering import cluster_skus


def create_agent(env, args, config, logger):
    """
    Create an agent based on command line arguments.
    
    Args:
        env: Environment instance
        args: Command line arguments
        config: Configuration
        logger: Logger instance
        
    Returns:
        Created agent
    """
    # Get dimensions from environment
    inventory_dim, forecast_dim, demand_dim = env.get_feature_dims()
    feature_dim = inventory_dim + forecast_dim + demand_dim
    action_size = 7  # [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0] × forecast
    
    # Create clustered or regular agent
    if args.clustered:
        logger.info(f"Creating clustered linear agent with {args.num_clusters} clusters")
        
        agent = ClusteredLinearAgent(
            num_clusters=args.num_clusters,
            feature_dim=feature_dim,
            action_size=action_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma
        )
        
        # Generate cluster mapping
        logger.info("Clustering SKUs...")
        cluster_mapping = cluster_skus(
            forecast_data=env.forecast_data,
            inventory_data=env.inventory_data,
            num_clusters=args.num_clusters,
            logger=logger
        )
        
        # Set cluster mapping in agent
        agent.set_cluster_mapping(cluster_mapping)
        
        # Save cluster mapping
        mapping_path = os.path.join(args.output_dir, "cluster_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump({str(k): int(v) for k, v in cluster_mapping.items()}, f)
        logger.info(f"Cluster mapping saved to {mapping_path}")
        
    else:
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
    
    return agent


def load_data(args, logger):
    """
    Load data from files.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        Tuple of (inventory_data, forecast_data, price_data, promo_data, weather_data)
    """
    # Initialize data containers
    inventory_data = None
    forecast_data = None
    price_data = {}
    promo_data = {}
    weather_data = {}
    
    # Load inventory data
    logger.info(f"Loading inventory data from {args.inventory_file}")
    inventory_data = pd.read_csv(args.inventory_file)
    
    # Standardize column names
    if 'product_id' in inventory_data.columns:
        inventory_data = inventory_data.rename(columns={'product_id': 'sku_id'})
    if 'on_hand_qty' in inventory_data.columns:
        inventory_data = inventory_data.rename(columns={'on_hand_qty': 'current_inventory'})
    
    # Load forecast data
    logger.info(f"Loading forecast data from {args.forecast_file}")
    forecast_data = pd.read_csv(args.forecast_file)
    
    # Standardize column names
    if 'product_id' in forecast_data.columns:
        forecast_data = forecast_data.rename(columns={'product_id': 'sku_id'})
    
    # Load optional data files
    if args.price_file and os.path.exists(args.price_file):
        logger.info(f"Loading price data from {args.price_file}")
        price_df = pd.read_csv(args.price_file)
        
        # Standardize column names
        if 'product_id' in price_df.columns:
            price_df = price_df.rename(columns={'product_id': 'sku_id'})
        if 'date' in price_df.columns:
            price_df = price_df.rename(columns={'date': 'price_date'})
        
        # Convert date to datetime
        price_df['price_date'] = pd.to_datetime(price_df['price_date'])
        
        # Create price dictionary
        for _, row in price_df.iterrows():
            price_data[(row['sku_id'], row['price_date'].to_pydatetime())] = row['price']
    
    if args.promo_file and os.path.exists(args.promo_file):
        logger.info(f"Loading promotion data from {args.promo_file}")
        promo_df = pd.read_csv(args.promo_file)
        
        # Standardize column names
        if 'product_id' in promo_df.columns:
            promo_df = promo_df.rename(columns={'product_id': 'sku_id'})
        if 'date' in promo_df.columns:
            promo_df = promo_df.rename(columns={'date': 'promo_date'})
        
        # Convert date to datetime
        promo_df['promo_date'] = pd.to_datetime(promo_df['promo_date'])
        
        # Create promotion dictionary
        for _, row in promo_df.iterrows():
            promo_info = {
                'type': row.get('promo_type', 0),
                'discount': row.get('discount', 0.0)
            }
            promo_data[(row['sku_id'], row['promo_date'].to_pydatetime())] = promo_info
    
    if args.weather_file and os.path.exists(args.weather_file):
        logger.info(f"Loading weather data from {args.weather_file}")
        weather_df = pd.read_csv(args.weather_file)
        
        # Convert date to datetime
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Create weather dictionary
        for _, row in weather_df.iterrows():
            weather_info = {
                'high_temp': row.get('high_temp', 70),
                'low_temp': row.get('low_temp', 50),
                'precip_prob': row.get('precip_prob', 0.0)
            }
            weather_data[row['date'].to_pydatetime()] = weather_info
    
    return inventory_data, forecast_data, price_data, promo_data, weather_data


def main():
    """Main function to run the inventory management system."""
    parser = argparse.ArgumentParser(description='Inventory Management System')
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], 
                       default='train', help='Operation mode')
    
    # Data options
    parser.add_argument('--inventory-file', required=True, help='Path to inventory data CSV')
    parser.add_argument('--forecast-file', required=True, help='Path to forecast data CSV')
    parser.add_argument('--price-file', help='Path to price data CSV')
    parser.add_argument('--promo-file', help='Path to promotion data CSV')
    parser.add_argument('--weather-file', help='Path to weather data CSV')
    
    # Agent options
    parser.add_argument('--clustered', action='store_true', help='Use clustered linear agent')
    parser.add_argument('--num-clusters', type=int, default=20, help='Number of clusters for clustered agent')
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
    
    # Reward options
    parser.add_argument('--oos-factor', type=float, default=3.0, help='Out-of-stock penalty factor')
    parser.add_argument('--waste-factor', type=float, default=1.0, help='Waste penalty factor')
    
    # Model options
    parser.add_argument('--model-path', help='Path to saved model')
    parser.add_argument('--output-dir', default='output', help='Directory for outputs')
    
    # Config options
    parser.add_argument('--config-file', help='Path to configuration file')
    parser.add_argument('--random-seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set up logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("main", log_dir=args.output_dir)
    
    # Load configuration
    config = Config()
    if args.config_file and os.path.exists(args.config_file):
        config = Config.load(args.config_file)
        logger.info(f"Loaded configuration from {args.config_file}")
    
    # Update reward config from args
    config.reward = RewardConfig(
        oos_penalty_factor=args.oos_factor,
        waste_penalty_factor=args.waste_factor
    )
    
    # Set random seed
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        logger.info(f"Random seed set to {args.random_seed}")
    
    # Load data
    inventory_data, forecast_data, price_data, promo_data, weather_data = load_data(args, logger)
    
    # Create environment
    logger.info("Creating environment")
    env = InventoryEnvironment(
        inventory_data=inventory_data,
        forecast_data=forecast_data,
        price_data=price_data,
        promotion_data=promo_data,
        weather_data=weather_data,
        env_config=config.environment,
        reward_config=config.reward
    )
    
    # Display environment stats
    logger.info(f"Environment created with {len(env.skus)} SKUs")
    
    # Run in appropriate mode
    if args.mode == 'train':
        # Create agent
        agent = create_agent(env, args, config, logger)
        
        # Create trainer
        trainer = Trainer(
            agent=agent,
            environment=env,
            output_dir=args.output_dir,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            save_every=args.save_every,
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
        logger.info(f"  Average Service Level (last 100): {np.mean(metrics['service_levels'][-100:]):.4f}")
        logger.info(f"  Average Waste Percentage (last 100): {np.mean(metrics['waste_percentages'][-100:]):.4f}")
        
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
        
        if args.clustered:
            agent = ClusteredLinearAgent.load(args.model_path)
        else:
            agent = LinearAgent.load(args.model_path)
        
        # Create trainer for evaluation
        trainer = Trainer(
            agent=agent,
            environment=env,
            output_dir=args.output_dir,
            logger=logger
        )
        
        # Evaluate the agent
        logger.info("Starting evaluation")
        eval_metrics = trainer.evaluate(num_episodes=10)
        
        # Log results
        logger.info(f"Evaluation completed")
        logger.info(f"Metrics:")
        logger.info(f"  Average Score: {eval_metrics['avg_score']:.2f}")
        logger.info(f"  Average Service Level: {eval_metrics['avg_service_level']:.4f}")
        logger.info(f"  Average Waste Percentage: {eval_metrics['avg_waste_percentage']:.4f}")
        
    elif args.mode == 'predict':
        # Check if model path is provided
        if not args.model_path:
            logger.error("Model path must be provided for prediction mode")
            return
        
        if not os.path.exists(args.model_path):
            logger.error(f"Model file {args.model_path} not found")
            return
        
        # Load agent
        logger.info(f"Loading model from {args.model_path}")
        
        if args.clustered:
            agent = ClusteredLinearAgent.load(args.model_path)
        else:
            agent = LinearAgent.load(args.model_path)
        
        # Create trainer for prediction
        trainer = Trainer(
            agent=agent,
            environment=env,
            output_dir=args.output_dir,
            logger=logger
        )
        
        # Generate predictions
        logger.info("Generating order predictions")
        predictions = trainer.predict_orders(num_days=14)
        
        # Save predictions
        predictions_path = os.path.join(args.output_dir, "order_predictions.csv")
        predictions.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")
        
        # Log summary
        logger.info(f"Prediction summary:")
        logger.info(f"  Total SKUs: {predictions['sku_id'].nunique()}")
        logger.info(f"  Total days: {predictions['day'].nunique()}")
        logger.info(f"  Total order quantity: {predictions['order_quantity'].sum()}")
        
        # Create summary plots
        plt.figure(figsize=(12, 10))
        
        # Average order quantity by day
        plt.subplot(2, 2, 1)
        day_orders = predictions.groupby('day')['order_quantity'].mean()
        plt.bar(day_orders.index, day_orders.values)
        plt.title('Average Order Quantity by Day')
        plt.xlabel('Day')
        plt.ylabel('Quantity')
        
        # Action distribution
        plt.subplot(2, 2, 2)
        action_counts = predictions['action_idx'].value_counts().sort_index()
        action_labels = ['0x', '0.5x', '0.8x', '1.0x', '1.2x', '1.5x', '2.0x']
        plt.bar([action_labels[i] for i in action_counts.index], action_counts.values)
        plt.title('Action Distribution')
        plt.xlabel('Action Multiplier')
        plt.ylabel('Count')
        
        # Order quantity vs forecast
        plt.subplot(2, 2, 3)
        plt.scatter(predictions['blended_forecast'], predictions['order_quantity'], alpha=0.3)
        plt.title('Order Quantity vs Forecast')
        plt.xlabel('Blended Forecast')
        plt.ylabel('Order Quantity')
        plt.grid(True, alpha=0.3)
        
        # Lead time distribution
        plt.subplot(2, 2, 4)
        lead_time_counts = predictions['lead_time'].value_counts().sort_index()
        plt.bar(lead_time_counts.index, lead_time_counts.values)
        plt.title('Lead Time Distribution')
        plt.xlabel('Lead Time (days)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "prediction_summary.png"))
        logger.info(f"Summary plots saved to {os.path.join(args.output_dir, 'prediction_summary.png')}")
    
    logger.info("Operation completed successfully")


if __name__ == "__main__":
    main()