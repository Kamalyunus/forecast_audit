"""
Example usage of Linear Function Approximation for forecast adjustment.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import logging

# Import forecast adjustment components
from forecast_environment import ForecastEnvironment
from linear_agent import LinearAgent, ClusteredLinearAgent
from trainer import ForecastAdjustmentTrainer
from clustering import cluster_skus
from generate_example_data import generate_forecast_data


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("Example")


def generate_sample_data(num_skus=100):
    """Generate sample forecast and historical data."""
    logger = logging.getLogger("DataGeneration")
    logger.info(f"Generating sample data with {num_skus} SKUs")
    
    # Generate forecast data
    forecast_data = generate_forecast_data(num_skus=num_skus, forecast_days=14)
    
    # Generate synthetic historical data
    historical_records = []
    
    for i in range(num_skus):
        sku_id = f"SKU_{i:04d}"
        
        # Extract forecast pattern for this SKU
        sku_forecasts = forecast_data[forecast_data['sku_id'] == sku_id]
        ml_cols = [col for col in sku_forecasts.columns if col.startswith('ml_day_')]
        
        if not sku_forecasts.empty:
            ml_values = sku_forecasts[ml_cols].iloc[0].values
            # Add noise and bias to create "actual" historical values
            mape = sku_forecasts['ml_mape_30d'].iloc[0]
            bias = sku_forecasts['ml_bias_30d'].iloc[0]
            
            # Create historical data for the last 60 days
            for day in range(60):
                # Use cyclic pattern from forecast with slight variations
                base_pattern_idx = day % len(ml_values)
                base_value = ml_values[base_pattern_idx]
                
                # Apply bias and random noise
                biased_value = base_value * (1 + bias)
                noise_range = biased_value * mape
                actual_value = max(0, biased_value + np.random.uniform(-noise_range, noise_range))
                
                # Use ISO format date to avoid parsing issues
                sale_date = f"2023-01-{day+1:02d}"
                
                historical_records.append({
                    'sku_id': sku_id,
                    'sale_date': sale_date,
                    'quantity': int(actual_value)
                })
    
    # Create historical data DataFrame
    historical_data = pd.DataFrame(historical_records)
    logger.info(f"Generated {len(historical_data)} historical records for {num_skus} SKUs")
    
    return forecast_data, historical_data


def example_simple_agent():
    """Example of training a simple linear agent for forecast adjustment."""
    logger = setup_logging()
    logger.info("Running example with simple linear agent for forecast adjustment")
    
    # Generate sample data
    logger.info("Generating sample data")
    forecast_data, historical_data = generate_sample_data(num_skus=50)
    
    # Create output directory
    output_dir = "example_output/forecast_adjustment"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save sample data
    forecast_data.to_csv(f"{output_dir}/sample_forecast_data.csv", index=False)
    historical_data.to_csv(f"{output_dir}/sample_historical_data.csv", index=False)
    
    # Create environment
    env = ForecastEnvironment(
        forecast_data=forecast_data,
        historical_data=historical_data,
        validation_length=30,
        forecast_horizon=7,
        optimize_for="both"
    )
    
    # Get state dimensions
    forecast_dim, error_dim, feature_dim = env.get_feature_dims()
    
    # Create linear agent
    agent = LinearAgent(
        feature_dim=feature_dim,
        action_size=7,  # [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3] 
        learning_rate=0.01,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # Create trainer
    trainer = ForecastAdjustmentTrainer(
        agent=agent,
        environment=env,
        output_dir=output_dir,
        num_episodes=100,  # Reduced for example
        max_steps=14,
        batch_size=32,
        save_every=20
    )
    
    # Train the agent
    logger.info("Training forecast adjustment agent")
    start_time = time.time()
    metrics = trainer.train()
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Final metrics:")
    logger.info(f"  Average Score (last 50): {np.mean(metrics['scores'][-50:]):.2f}")
    logger.info(f"  Average MAPE Improvement (last 50): {np.mean(metrics['mape_improvements'][-50:]):.4f}")
    logger.info(f"  Average Bias Improvement (last 50): {np.mean(metrics['bias_improvements'][-50:]):.4f}")
    
    # Evaluate the agent
    logger.info("Evaluating forecast adjustment agent")
    eval_metrics = trainer.evaluate(num_episodes=10)
    
    logger.info(f"Evaluation metrics:")
    logger.info(f"  Average Score: {eval_metrics['avg_score']:.2f}")
    logger.info(f"  Average MAPE Improvement: {eval_metrics['avg_mape_improvement']:.4f}")
    logger.info(f"  Average Bias Improvement: {eval_metrics['avg_bias_improvement']:.4f}")
    
    # Generate adjusted forecasts
    logger.info("Generating adjusted forecasts")
    adjustments = trainer.generate_adjusted_forecasts(num_days=14)
    
    # Save adjustments
    adjustments_path = os.path.join(output_dir, "adjusted_forecasts.csv")
    adjustments.to_csv(adjustments_path, index=False)
    
    logger.info(f"Adjusted forecasts saved to {adjustments_path}")
    logger.info(f"Example completed successfully")
    
    return agent, env, trainer


def optimize_for_mape():
    """Example of optimizing specifically for MAPE improvement."""
    logger = setup_logging()
    logger.info("Running example optimizing specifically for MAPE")
    
    # Generate sample data
    logger.info("Generating sample data")
    forecast_data, historical_data = generate_sample_data(num_skus=100)
    
    # Create output directories
    mape_dir = "example_output/mape_optimization"
    bias_dir = "example_output/bias_optimization"
    os.makedirs(mape_dir, exist_ok=True)
    os.makedirs(bias_dir, exist_ok=True)
    
    # Create environment optimized for MAPE
    mape_env = ForecastEnvironment(
        forecast_data=forecast_data,
        historical_data=historical_data,
        validation_length=30,
        forecast_horizon=7,
        optimize_for="mape"  # Focus on MAPE
    )
    
    # Create environment optimized for bias
    bias_env = ForecastEnvironment(
        forecast_data=forecast_data,
        historical_data=historical_data,
        validation_length=30,
        forecast_horizon=7,
        optimize_for="bias"  # Focus on bias
    )
    
    # Get dimensions
    forecast_dim, error_dim, feature_dim = mape_env.get_feature_dims()
    
    # Create agents
    mape_agent = LinearAgent(
        feature_dim=feature_dim,
        action_size=7,
        learning_rate=0.01,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    bias_agent = LinearAgent(
        feature_dim=feature_dim,
        action_size=7,
        learning_rate=0.01,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # Create trainers
    mape_trainer = ForecastAdjustmentTrainer(
        agent=mape_agent,
        environment=mape_env,
        output_dir=mape_dir,
        num_episodes=100,
        max_steps=14,
        batch_size=32,
        save_every=20,
        optimize_for="mape"
    )
    
    bias_trainer = ForecastAdjustmentTrainer(
        agent=bias_agent,
        environment=bias_env,
        output_dir=bias_dir,
        num_episodes=100,
        max_steps=14,
        batch_size=32,
        save_every=20,
        optimize_for="bias"
    )
    
    # Train both agents
    logger.info("Training MAPE-optimized agent")
    mape_metrics = mape_trainer.train()
    
    logger.info("Training Bias-optimized agent")
    bias_metrics = bias_trainer.train()
    
    # Evaluate both agents
    logger.info("Evaluating MAPE-optimized agent")
    mape_eval = mape_trainer.evaluate(num_episodes=10)
    
    logger.info("Evaluating Bias-optimized agent")
    bias_eval = bias_trainer.evaluate(num_episodes=10)
    
    # Compare results
    logger.info(f"Comparison of optimization strategies:")
    logger.info(f"  MAPE-optimized: MAPE Imp = {mape_eval['avg_mape_improvement']:.4f}, Bias Imp = {mape_eval['avg_bias_improvement']:.4f}")
    logger.info(f"  Bias-optimized: MAPE Imp = {bias_eval['avg_mape_improvement']:.4f}, Bias Imp = {bias_eval['avg_bias_improvement']:.4f}")
    
    # Generate forecasts from both agents
    mape_forecasts = mape_trainer.generate_adjusted_forecasts(num_days=14)
    bias_forecasts = bias_trainer.generate_adjusted_forecasts(num_days=14)
    
    # Save forecasts
    mape_forecasts.to_csv(os.path.join(mape_dir, "mape_adjusted_forecasts.csv"), index=False)
    bias_forecasts.to_csv(os.path.join(bias_dir, "bias_adjusted_forecasts.csv"), index=False)
    
    # Create comparative visualization
    plt.figure(figsize=(15, 10))
    
    # Training progress comparison (MAPE improvement)
    plt.subplot(2, 2, 1)
    plt.plot(mape_metrics['mape_improvements'], 'b-', label='MAPE-optimized')
    plt.plot(bias_metrics['mape_improvements'], 'r-', label='Bias-optimized')
    plt.title('MAPE Improvement During Training')
    plt.xlabel('Episode')
    plt.ylabel('Improvement Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training progress comparison (Bias improvement)
    plt.subplot(2, 2, 2)
    plt.plot(mape_metrics['bias_improvements'], 'b-', label='MAPE-optimized')
    plt.plot(bias_metrics['bias_improvements'], 'r-', label='Bias-optimized')
    plt.title('Bias Improvement During Training')
    plt.xlabel('Episode')
    plt.ylabel('Improvement Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Action distribution comparison
    plt.subplot(2, 2, 3)
    
    # MAPE-optimized agent actions
    mape_actions = mape_agent.action_counts
    mape_total = np.sum(mape_actions)
    mape_dist = mape_actions / mape_total if mape_total > 0 else np.zeros_like(mape_actions)
    
    # Bias-optimized agent actions
    bias_actions = bias_agent.action_counts
    bias_total = np.sum(bias_actions)
    bias_dist = bias_actions / bias_total if bias_total > 0 else np.zeros_like(bias_actions)
    
    # Plot comparison
    adjustment_factors = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    x = np.arange(len(adjustment_factors))
    width = 0.35
    
    plt.bar(x - width/2, mape_dist, width, label='MAPE-optimized')
    plt.bar(x + width/2, bias_dist, width, label='Bias-optimized')
    
    plt.xlabel('Adjustment Factor')
    plt.ylabel('Frequency')
    plt.title('Action Distribution Comparison')
    plt.xticks(x, [f"{f:.1f}x" for f in adjustment_factors])
    plt.legend()
    
    # Final performance comparison
    plt.subplot(2, 2, 4)
    metrics = ['MAPE Improvement', 'Bias Improvement']
    mape_values = [mape_eval['avg_mape_improvement'], mape_eval['avg_bias_improvement']]
    bias_values = [bias_eval['avg_mape_improvement'], bias_eval['avg_bias_improvement']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, mape_values, width, label='MAPE-optimized')
    plt.bar(x + width/2, bias_values, width, label='Bias-optimized')
    
    plt.xlabel('Metric')
    plt.ylabel('Improvement Ratio')
    plt.title('Final Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("example_output/optimization_comparison.png")
    logger.info("Comparative visualization saved to example_output/optimization_comparison.png")
    
    logger.info("Optimization comparison completed successfully")
    
    return mape_agent, bias_agent


if __name__ == "__main__":
    print("Forecast Adjustment Examples")
    print("1. Train simple linear agent for forecast adjustment")
    print("2. Compare MAPE vs Bias optimization strategies")
    
    choice = input("Enter your choice (1-2): ")
    
    if choice == '1':
        example_simple_agent()
    elif choice == '2':
        optimize_for_mape()
    else:
        print("Invalid choice!")