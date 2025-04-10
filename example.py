"""
Example usage of Linear Function Approximation for inventory management.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import logging

from inventory_rl.data.data_generator import generate_sample_data
from inventory_rl.environment.inventory_env import InventoryEnvironment
from inventory_rl.utils.config import Config, RewardConfig

# Import linear agent implementation
from linear_agent import LinearAgent, ClusteredLinearAgent
from trainer import Trainer
from clustering import cluster_skus


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("Example")


def example_simple_agent():
    """Example of training a simple linear agent."""
    logger = setup_logging()
    logger.info("Running example with simple linear agent")
    
    # Generate sample data
    logger.info("Generating sample data")
    inventory_data, forecast_data, historical_data, price_data, promo_data, weather_data = generate_sample_data(
        num_skus=50  # Use a small number of SKUs for the example
    )
    
    # Create configuration
    config = Config()
    config.reward = RewardConfig(
        oos_penalty_factor=3.0,  # Higher penalty for stockouts
        waste_penalty_factor=1.0
    )
    
    # Create environment
    env = InventoryEnvironment(
        inventory_data=inventory_data,
        forecast_data=forecast_data,
        price_data=price_data,
        promotion_data=promo_data,
        weather_data=weather_data,
        env_config=config.environment,
        reward_config=config.reward
    )
    
    # Get state dimensions
    inventory_dim, forecast_dim, demand_dim = env.get_feature_dims()
    feature_dim = inventory_dim + forecast_dim + demand_dim
    
    # Create linear agent
    agent = LinearAgent(
        feature_dim=feature_dim,
        action_size=7,  # [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0] × forecast
        learning_rate=0.01,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # Create output directory
    output_dir = "example_output/simple"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        agent=agent,
        environment=env,
        output_dir=output_dir,
        num_episodes=200,  # Reduced for example
        max_steps=14,
        batch_size=32,
        save_every=50
    )
    
    # Train the agent
    logger.info("Training linear agent")
    start_time = time.time()
    metrics = trainer.train()
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Final metrics:")
    logger.info(f"  Average Score (last 100): {np.mean(metrics['scores'][-100:]):.2f}")
    logger.info(f"  Average Service Level (last 100): {np.mean(metrics['service_levels'][-100:]):.4f}")
    logger.info(f"  Average Waste % (last 100): {np.mean(metrics['waste_percentages'][-100:]):.4f}")
    
    # Evaluate the agent
    logger.info("Evaluating agent")
    eval_metrics = trainer.evaluate(num_episodes=10)
    
    logger.info(f"Evaluation metrics:")
    logger.info(f"  Average Score: {eval_metrics['avg_score']:.2f}")
    logger.info(f"  Average Service Level: {eval_metrics['avg_service_level']:.4f}")
    logger.info(f"  Average Waste Percentage: {eval_metrics['avg_waste_percentage']:.4f}")
    
    # Generate predictions
    logger.info("Generating predictions")
    predictions = trainer.predict_orders(num_days=14)
    
    # Save predictions
    predictions_path = os.path.join(output_dir, "predictions.csv")
    predictions.to_csv(predictions_path, index=False)
    
    logger.info(f"Predictions saved to {predictions_path}")
    logger.info(f"Example completed successfully")
    
    return agent, env, trainer


def example_clustered_agent():
    """Example of training a clustered linear agent."""
    logger = setup_logging()
    logger.info("Running example with clustered linear agent")
    
    # Generate sample data with more SKUs
    logger.info("Generating sample data")
    inventory_data, forecast_data, historical_data, price_data, promo_data, weather_data = generate_sample_data(
        num_skus=200  # More SKUs to demonstrate clustering
    )
    
    # Create configuration
    config = Config()
    config.reward = RewardConfig(
        oos_penalty_factor=3.0,
        waste_penalty_factor=1.0
    )
    
    # Create environment
    env = InventoryEnvironment(
        inventory_data=inventory_data,
        forecast_data=forecast_data,
        price_data=price_data,
        promotion_data=promo_data,
        weather_data=weather_data,
        env_config=config.environment,
        reward_config=config.reward
    )
    
    # Get state dimensions
    inventory_dim, forecast_dim, demand_dim = env.get_feature_dims()
    feature_dim = inventory_dim + forecast_dim + demand_dim
    
    # Number of clusters
    num_clusters = 5  # Use 5 clusters for demonstration
    
    # Create clustered agent
    agent = ClusteredLinearAgent(
        num_clusters=num_clusters,
        feature_dim=feature_dim,
        action_size=7,  # [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0] × forecast
        learning_rate=0.01,
        gamma=0.99
    )
    
    # Create cluster mapping
    logger.info("Clustering SKUs")
    cluster_mapping = cluster_skus(
        forecast_data=forecast_data,
        inventory_data=inventory_data,
        historical_data=historical_data,
        num_clusters=num_clusters,
        logger=logger
    )
    
    # Set cluster mapping in agent
    agent.set_cluster_mapping(cluster_mapping)
    
    # Create output directory
    output_dir = "example_output/clustered"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cluster mapping
    mapping_path = os.path.join(output_dir, "cluster_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump({str(k): int(v) for k, v in cluster_mapping.items()}, f)
    
    # Create trainer
    trainer = Trainer(
        agent=agent,
        environment=env,
        output_dir=output_dir,
        num_episodes=200,  # Reduced for example
        max_steps=14,
        batch_size=32,
        save_every=50
    )
    
    # Train the agent
    logger.info("Training clustered linear agent")
    start_time = time.time()
    metrics = trainer.train()
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Final metrics:")
    logger.info(f"  Average Score (last 100): {np.mean(metrics['scores'][-100:]):.2f}")
    logger.info(f"  Average Service Level (last 100): {np.mean(metrics['service_levels'][-100:]):.4f}")
    logger.info(f"  Average Waste % (last 100): {np.mean(metrics['waste_percentages'][-100:]):.4f}")
    
    # Evaluate the agent
    logger.info("Evaluating agent")
    eval_metrics = trainer.evaluate(num_episodes=10)
    
    logger.info(f"Evaluation metrics:")
    logger.info(f"  Average Score: {eval_metrics['avg_score']:.2f}")
    logger.info(f"  Average Service Level: {eval_metrics['avg_service_level']:.4f}")
    logger.info(f"  Average Waste Percentage: {eval_metrics['avg_waste_percentage']:.4f}")
    
    # Analyze cluster-specific performance
    logger.info("Cluster-specific action distributions:")
    for cluster_id in range(num_clusters):
        cluster_agent = agent.agents[cluster_id]
        action_counts = cluster_agent.action_counts
        total_actions = np.sum(action_counts)
        if total_actions > 0:
            action_distribution = action_counts / total_actions
            logger.info(f"  Cluster {cluster_id}: {action_distribution.round(3)}")
    
    # Generate predictions
    logger.info("Generating predictions")
    predictions = trainer.predict_orders(num_days=14)
    
    # Save predictions
    predictions_path = os.path.join(output_dir, "predictions.csv")
    predictions.to_csv(predictions_path, index=False)
    
    logger.info(f"Predictions saved to {predictions_path}")
    logger.info(f"Example completed successfully")
    
    return agent, env, trainer


def performance_benchmark():
    """Benchmark performance with increasing number of SKUs."""
    logger = setup_logging()
    logger.info("Running performance benchmark")
    
    # SKU counts to test
    sku_counts = [100, 500, 1000, 5000]
    
    # Results storage
    results = []
    
    for num_skus in sku_counts:
        logger.info(f"Testing with {num_skus} SKUs")
        
        # Generate sample data
        inventory_data, forecast_data, _, price_data, promo_data, weather_data = generate_sample_data(
            num_skus=num_skus
        )
        
        # Create environment
        config = Config()
        env = InventoryEnvironment(
            inventory_data=inventory_data,
            forecast_data=forecast_data,
            price_data=price_data,
            promotion_data=promo_data,
            weather_data=weather_data,
            env_config=config.environment,
            reward_config=config.reward
        )
        
        # Get dimensions
        inventory_dim, forecast_dim, demand_dim = env.get_feature_dims()
        feature_dim = inventory_dim + forecast_dim + demand_dim
        
        # Create output directory
        output_dir = f"benchmark_output/{num_skus}_skus"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test simple agent
        logger.info("Testing simple agent")
        simple_agent = LinearAgent(
            feature_dim=feature_dim,
            action_size=7
        )
        
        simple_trainer = Trainer(
            agent=simple_agent,
            environment=env,
            output_dir=output_dir,
            num_episodes=5  # Just for benchmarking
        )
        
        # Measure training time
        start_time = time.time()
        simple_trainer.train(verbose=False)
        simple_train_time = time.time() - start_time
        
        # Measure prediction time
        start_time = time.time()
        simple_trainer.predict_orders(num_days=7)
        simple_predict_time = time.time() - start_time
        
        # Test clustered agent
        logger.info("Testing clustered agent")
        num_clusters = min(20, num_skus // 50 + 1)  # Simple heuristic
        
        clustered_agent = ClusteredLinearAgent(
            num_clusters=num_clusters,
            feature_dim=feature_dim,
            action_size=7
        )
        
        # Simple clustering
        cluster_mapping = {sku: i % num_clusters for i, sku in enumerate(env.skus)}
        clustered_agent.set_cluster_mapping(cluster_mapping)
        
        clustered_trainer = Trainer(
            agent=clustered_agent,
            environment=env,
            output_dir=output_dir,
            num_episodes=5  # Just for benchmarking
        )
        
        # Measure training time
        start_time = time.time()
        clustered_trainer.train(verbose=False)
        clustered_train_time = time.time() - start_time
        
        # Measure prediction time
        start_time = time.time()
        clustered_trainer.predict_orders(num_days=7)
        clustered_predict_time = time.time() - start_time
        
        # Store results
        results.append({
            'num_skus': num_skus,
            'simple_train_time': simple_train_time,
            'simple_predict_time': simple_predict_time,
            'clustered_train_time': clustered_train_time,
            'clustered_predict_time': clustered_predict_time,
            'num_clusters': num_clusters
        })
        
        logger.info(f"Results for {num_skus} SKUs:")
        logger.info(f"  Simple Agent - Train: {simple_train_time:.2f}s, Predict: {simple_predict_time:.2f}s")
        logger.info(f"  Clustered Agent ({num_clusters} clusters) - Train: {clustered_train_time:.2f}s, Predict: {clustered_predict_time:.2f}s")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    benchmark_dir = "benchmark_output"
    os.makedirs(benchmark_dir, exist_ok=True)
    results_df.to_csv(os.path.join(benchmark_dir, "benchmark_results.csv"), index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Training time
    plt.subplot(2, 2, 1)
    plt.plot(results_df['num_skus'], results_df['simple_train_time'], 'bo-', label='Simple Agent')
    plt.plot(results_df['num_skus'], results_df['clustered_train_time'], 'ro-', label='Clustered Agent')
    plt.title('Training Time Scaling')
    plt.xlabel('Number of SKUs')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Prediction time
    plt.subplot(2, 2, 2)
    plt.plot(results_df['num_skus'], results_df['simple_predict_time'], 'bo-', label='Simple Agent')
    plt.plot(results_df['num_skus'], results_df['clustered_predict_time'], 'ro-', label='Clustered Agent')
    plt.title('Prediction Time Scaling')
    plt.xlabel('Number of SKUs')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training time per SKU
    plt.subplot(2, 2, 3)
    simple_train_per_sku = results_df['simple_train_time'] / results_df['num_skus']
    clustered_train_per_sku = results_df['clustered_train_time'] / results_df['num_skus']
    plt.plot(results_df['num_skus'], simple_train_per_sku * 1000, 'bo-', label='Simple Agent')
    plt.plot(results_df['num_skus'], clustered_train_per_sku * 1000, 'ro-', label='Clustered Agent')
    plt.title('Training Time per SKU')
    plt.xlabel('Number of SKUs')
    plt.ylabel('Time (milliseconds per SKU)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Prediction time per SKU
    plt.subplot(2, 2, 4)
    simple_predict_per_sku = results_df['simple_predict_time'] / results_df['num_skus']
    clustered_predict_per_sku = results_df['clustered_predict_time'] / results_df['num_skus']
    plt.plot(results_df['num_skus'], simple_predict_per_sku * 1000, 'bo-', label='Simple Agent')
    plt.plot(results_df['num_skus'], clustered_predict_per_sku * 1000, 'ro-', label='Clustered Agent')
    plt.title('Prediction Time per SKU')
    plt.xlabel('Number of SKUs')
    plt.ylabel('Time (milliseconds per SKU)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(benchmark_dir, "benchmark_results.png"))
    
    logger.info(f"Benchmark results saved to {benchmark_dir}")
    
    return results_df


if __name__ == "__main__":
    print("Linear Function Approximation Examples")
    print("1. Train simple linear agent")
    print("2. Train clustered linear agent")
    print("3. Run performance benchmark")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        example_simple_agent()
    elif choice == '2':
        example_clustered_agent()
    elif choice == '3':
        performance_benchmark()
    else:
        print("Invalid choice!")