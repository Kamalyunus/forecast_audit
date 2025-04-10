"""
Training and evaluation logic for Linear Agent inventory management.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm


class Trainer:
    """
    Trainer for Linear Agent inventory management.
    """
    
    def __init__(self, 
                agent,
                environment, 
                output_dir: str = "output",
                num_episodes: int = 500,
                max_steps: int = 14,
                batch_size: int = 32,
                save_every: int = 50,
                logger: Optional[logging.Logger] = None):
        """
        Initialize trainer.
        
        Args:
            agent: LinearAgent or ClusteredLinearAgent
            environment: Inventory environment
            output_dir: Directory for outputs
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            batch_size: Batch size for updates
            save_every: How often to save the model
            logger: Logger instance
        """
        self.agent = agent
        self.env = environment
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.save_every = save_every
        
        # Set up logger
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("Trainer")
        else:
            self.logger = logger
            
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.model_dir = os.path.join(output_dir, "models")
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Metrics
        self.scores = []
        self.service_levels = []
        self.waste_percentages = []
        self.training_time = 0
        
    def train(self, verbose: bool = True) -> Dict:
        """
        Train the linear agent.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary of training metrics
        """
        self.logger.info(f"Starting training for {self.num_episodes} episodes")
        start_time = time.time()
        
        # Track best metrics for model saving
        best_score = float('-inf')
        best_service_level = 0.0
        
        # Training loop
        for episode in tqdm(range(1, self.num_episodes + 1), disable=not verbose):
            state = self.env.reset()
            episode_score = 0
            episode_td_errors = []
            
            # Metrics for this episode
            sku_sales = {sku: 0 for sku in self.env.skus}
            sku_stockouts = {sku: 0 for sku in self.env.skus}
            sku_waste = {sku: 0 for sku in self.env.skus}
            
            for step in range(self.max_steps):
                actions = {}
                
                # Determine actions for all SKUs
                for i, sku in enumerate(self.env.skus):
                    # Extract state components
                    inventory_dim, forecast_dim, demand_dim = self.env.get_feature_dims()
                    
                    # State features for this SKU
                    sku_state = state[i]
                    
                    # Get action from agent
                    if hasattr(self.agent, 'get_cluster_id'):
                        # Clustered agent
                        action_idx = self.agent.act(sku_state, sku)
                    else:
                        # Regular agent
                        action_idx = self.agent.act(sku_state)
                    
                    # Calculate order quantity based on forecast
                    # Extract forecasts
                    forecast_features = sku_state[inventory_dim:inventory_dim+forecast_dim]
                    lead_time = np.random.randint(2, 4)  # 2-3 day lead time
                    forecast_idx = min(lead_time, forecast_dim // 2 - 1)
                    
                    # Get ML and ARIMA forecasts
                    ml_forecast = forecast_features[forecast_idx]
                    arima_forecast = forecast_features[forecast_idx + forecast_dim // 2]
                    
                    # Blend forecasts (simple average)
                    blended_forecast = (ml_forecast + arima_forecast) / 2
                    
                    # Calculate order using action multipliers
                    action_multipliers = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
                    order_quantity = int(blended_forecast * action_multipliers[action_idx])
                    actions[sku] = order_quantity
                
                # Take step in environment
                next_state, rewards, done, info = self.env.step(actions)
                
                # Update episode metrics
                episode_score += sum(rewards.values())
                
                # Update per-SKU metrics
                for sku in self.env.skus:
                    sku_sales[sku] += info['sales'][sku] - sku_sales[sku]
                    sku_stockouts[sku] += info['stockouts'][sku] - sku_stockouts[sku]
                    sku_waste[sku] += info['waste'][sku] - sku_waste[sku]
                
                # Update agent for each SKU
                for i, sku in enumerate(self.env.skus):
                    sku_state = state[i]
                    next_sku_state = next_state[i]
                    
                    # Determine action index that was taken
                    action_idx = np.argmax([
                        int(actions[sku] == 0), 
                        int(actions[sku] > 0 and actions[sku] < ml_forecast * 0.7),
                        int(actions[sku] >= ml_forecast * 0.7 and actions[sku] < ml_forecast * 0.9),
                        int(actions[sku] >= ml_forecast * 0.9 and actions[sku] < ml_forecast * 1.1),
                        int(actions[sku] >= ml_forecast * 1.1 and actions[sku] < ml_forecast * 1.3),
                        int(actions[sku] >= ml_forecast * 1.3 and actions[sku] < ml_forecast * 1.8),
                        int(actions[sku] >= ml_forecast * 1.8)
                    ])
                    
                    # Update agent
                    if hasattr(self.agent, 'update'):
                        # Direct agent
                        td_error = self.agent.update(sku_state, action_idx, rewards[sku], next_sku_state, done)
                        episode_td_errors.append(td_error)
                    elif hasattr(self.agent, 'get_cluster_id'):
                        # Clustered agent
                        td_error = self.agent.update(sku_state, action_idx, rewards[sku], next_sku_state, done, sku)
                        episode_td_errors.append(td_error)
                
                # Batch update (if available)
                if hasattr(self.agent, 'batch_update'):
                    self.agent.batch_update(self.batch_size)
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Calculate service level and waste percentage
            total_demand = sum(sku_sales.values()) + sum(sku_stockouts.values())
            service_level = sum(sku_sales.values()) / total_demand if total_demand > 0 else 0
            
            total_handled = sum(sku_sales.values()) + sum(sku_waste.values())
            waste_percentage = sum(sku_waste.values()) / total_handled if total_handled > 0 else 0
            
            # Store metrics
            self.scores.append(episode_score)
            self.service_levels.append(service_level)
            self.waste_percentages.append(waste_percentage)
            
            # Log progress
            if verbose and (episode % 10 == 0 or episode == 1):
                avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
                avg_service = np.mean(self.service_levels[-100:]) if len(self.service_levels) >= 100 else np.mean(self.service_levels)
                avg_waste = np.mean(self.waste_percentages[-100:]) if len(self.waste_percentages) >= 100 else np.mean(self.waste_percentages)
                avg_td_error = np.mean(episode_td_errors) if episode_td_errors else 0
                
                self.logger.info(f"Episode {episode}/{self.num_episodes} | "
                              f"Score: {episode_score:.2f} | "
                              f"Avg Score: {avg_score:.2f} | "
                              f"Service: {service_level:.4f} | "
                              f"Waste: {waste_percentage:.4f} | "
                              f"TD Error: {avg_td_error:.4f}")
            
            # Save models periodically
            if episode % self.save_every == 0:
                model_path = os.path.join(self.model_dir, f"model_episode_{episode}.pkl")
                self.agent.save(model_path)
                
                # Plot progress
                self._plot_training_progress()
            
            # Save best models
            if episode_score > best_score:
                best_score = episode_score
                best_model_path = os.path.join(self.model_dir, "best_score_model.pkl")
                self.agent.save(best_model_path)
                
            if service_level > best_service_level:
                best_service_level = service_level
                best_sl_model_path = os.path.join(self.model_dir, "best_service_model.pkl")
                self.agent.save(best_sl_model_path)
        
        # Final save
        final_model_path = os.path.join(self.model_dir, "final_model.pkl")
        self.agent.save(final_model_path)
        
        # Calculate training time
        self.training_time = time.time() - start_time
        self.logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        # Final plot
        self._plot_training_progress()
        
        # Return metrics
        metrics = {
            'scores': self.scores,
            'service_levels': self.service_levels,
            'waste_percentages': self.waste_percentages,
            'training_time': self.training_time,
            'best_score': best_score,
            'best_service_level': best_service_level,
            'final_model_path': final_model_path
        }
        
        return metrics
    
    def _plot_training_progress(self):
        """Create plot of training progress metrics."""
        plt.figure(figsize=(15, 10))
        
        # Plot scores
        plt.subplot(2, 2, 1)
        plt.plot(self.scores)
        plt.title('Training Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        # Moving average
        if len(self.scores) > 10:
            moving_avg = np.convolve(self.scores, np.ones(10)/10, mode='valid')
            plt.plot(range(9, len(self.scores)), moving_avg, 'r-')
        
        # Plot service level
        plt.subplot(2, 2, 2)
        plt.plot(self.service_levels)
        plt.title('Service Level')
        plt.xlabel('Episode')
        plt.ylabel('Service Level')
        plt.ylim([0, 1])
        
        # Plot waste percentage
        plt.subplot(2, 2, 3)
        plt.plot(self.waste_percentages)
        plt.title('Waste Percentage')
        plt.xlabel('Episode')
        plt.ylabel('Waste %')
        plt.ylim([0, 1])
        
        # Plot service level vs waste (last 100 episodes)
        plt.subplot(2, 2, 4)
        if len(self.service_levels) > 100:
            plt.scatter(self.service_levels[-100:], self.waste_percentages[-100:], alpha=0.7)
        else:
            plt.scatter(self.service_levels, self.waste_percentages, alpha=0.7)
        plt.title('Service Level vs Waste Trade-off')
        plt.xlabel('Service Level')
        plt.ylabel('Waste %')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'))
        plt.close()
    
    def evaluate(self, num_episodes: int = 10, verbose: bool = True) -> Dict:
        """
        Evaluate the linear agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            verbose: Whether to print progress
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Starting evaluation for {num_episodes} episodes")
        
        # Evaluation metrics
        eval_scores = []
        eval_service_levels = []
        eval_waste_percentages = []
        
        # Evaluation loop
        for episode in tqdm(range(1, num_episodes + 1), disable=not verbose):
            state = self.env.reset()
            episode_score = 0
            
            for step in range(self.max_steps):
                actions = {}
                
                # Determine actions for all SKUs (no exploration)
                for i, sku in enumerate(self.env.skus):
                    # Extract state components
                    inventory_dim, forecast_dim, demand_dim = self.env.get_feature_dims()
                    
                    # State features for this SKU
                    sku_state = state[i]
                    
                    # Get action from agent (no exploration)
                    if hasattr(self.agent, 'get_cluster_id'):
                        # Clustered agent
                        action_idx = self.agent.act(sku_state, sku, explore=False)
                    else:
                        # Regular agent
                        action_idx = self.agent.act(sku_state, explore=False)
                    
                    # Extract forecasts
                    forecast_features = sku_state[inventory_dim:inventory_dim+forecast_dim]
                    lead_time = min(step + 3, forecast_dim // 2 - 1)
                    
                    # Get ML and ARIMA forecasts
                    ml_forecast = forecast_features[lead_time]
                    arima_forecast = forecast_features[lead_time + forecast_dim // 2]
                    
                    # Blend forecasts
                    blended_forecast = (ml_forecast + arima_forecast) / 2
                    
                    # Calculate order using action multipliers
                    action_multipliers = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
                    order_quantity = int(blended_forecast * action_multipliers[action_idx])
                    actions[sku] = order_quantity
                
                # Take step in environment
                next_state, rewards, done, info = self.env.step(actions)
                
                # Update episode metrics
                episode_score += sum(rewards.values())
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Calculate service level and waste percentage
            total_demand = sum(info['sales'].values()) + sum(info['stockouts'].values())
            service_level = sum(info['sales'].values()) / total_demand if total_demand > 0 else 0
            
            total_handled = sum(info['sales'].values()) + sum(info['waste'].values())
            waste_percentage = sum(info['waste'].values()) / total_handled if total_handled > 0 else 0
            
            # Store metrics
            eval_scores.append(episode_score)
            eval_service_levels.append(service_level)
            eval_waste_percentages.append(waste_percentage)
            
            if verbose:
                self.logger.info(f"Eval Episode {episode}/{num_episodes} | "
                              f"Score: {episode_score:.2f} | "
                              f"Service: {service_level:.4f} | "
                              f"Waste: {waste_percentage:.4f}")
        
        # Calculate aggregate metrics
        avg_score = np.mean(eval_scores)
        avg_service_level = np.mean(eval_service_levels)
        avg_waste_percentage = np.mean(eval_waste_percentages)
        
        self.logger.info(f"Evaluation complete | "
                      f"Avg Score: {avg_score:.2f} | "
                      f"Avg Service: {avg_service_level:.4f} | "
                      f"Avg Waste: {avg_waste_percentage:.4f}")
        
        # Return metrics
        metrics = {
            'scores': eval_scores,
            'service_levels': eval_service_levels,
            'waste_percentages': eval_waste_percentages,
            'avg_score': avg_score,
            'avg_service_level': avg_service_level,
            'avg_waste_percentage': avg_waste_percentage
        }
        
        return metrics
    
    def predict_orders(self, num_days: int = 14) -> pd.DataFrame:
        """
        Generate order predictions using the linear agent.
        
        Args:
            num_days: Number of days to predict
            
        Returns:
            DataFrame of order predictions
        """
        self.logger.info(f"Generating order predictions for {num_days} days")
        
        # Reset environment
        state = self.env.reset()
        
        # Predictions storage
        predictions = []
        
        for day in range(min(num_days, self.max_steps)):
            # Process each SKU
            for i, sku in enumerate(self.env.skus):
                # Extract state components
                inventory_dim, forecast_dim, demand_dim = self.env.get_feature_dims()
                
                # State features for this SKU
                sku_state = state[i]
                
                # Get action from agent (no exploration)
                if hasattr(self.agent, 'get_cluster_id'):
                    # Clustered agent
                    action_idx = self.agent.act(sku_state, sku, explore=False)
                else:
                    # Regular agent
                    action_idx = self.agent.act(sku_state, explore=False)
                
                # For each lead time (typically 2-3 days)
                for lead_time in range(2, 4):
                    forecast_idx = min(day + lead_time, forecast_dim // 2 - 1)
                    
                    # Extract ML and ARIMA forecasts
                    forecast_features = sku_state[inventory_dim:inventory_dim+forecast_dim]
                    ml_forecast = forecast_features[forecast_idx]
                    arima_forecast = forecast_features[forecast_idx + forecast_dim // 2]
                    
                    # Blend forecasts
                    blended_forecast = (ml_forecast + arima_forecast) / 2
                    
                    # Calculate order quantity
                    action_multipliers = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
                    multiplier = action_multipliers[action_idx]
                    order_quantity = int(blended_forecast * multiplier)
                    
                    # Add to predictions
                    predictions.append({
                        'sku_id': sku,
                        'day': day,
                        'lead_time': lead_time,
                        'delivery_day': day + lead_time,
                        'ml_forecast': float(ml_forecast),
                        'arima_forecast': float(arima_forecast),
                        'blended_forecast': float(blended_forecast),
                        'action_idx': int(action_idx),
                        'multiplier': float(multiplier),
                        'order_quantity': order_quantity,
                        'current_inventory': float(sku_state[0] * 1000)
                    })
            
            # Calculate actions for environment step
            actions = {}
            for i, sku in enumerate(self.env.skus):
                # State features
                sku_state = state[i]
                
                # Get action
                if hasattr(self.agent, 'get_cluster_id'):
                    action_idx = self.agent.act(sku_state, sku, explore=False)
                else:
                    action_idx = self.agent.act(sku_state, explore=False)
                
                # Extract forecasts for current day
                forecast_features = sku_state[inventory_dim:inventory_dim+forecast_dim]
                forecast_idx = min(day, forecast_dim // 2 - 1)
                ml_forecast = forecast_features[forecast_idx]
                arima_forecast = forecast_features[forecast_idx + forecast_dim // 2]
                blended_forecast = (ml_forecast + arima_forecast) / 2
                
                # Calculate order
                action_multipliers = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
                order_quantity = int(blended_forecast * action_multipliers[action_idx])
                actions[sku] = order_quantity
            
            # Take environment step
            next_state, _, done, _ = self.env.step(actions)
            state = next_state
            
            if done:
                break
        
        # Convert to DataFrame
        return pd.DataFrame(predictions)