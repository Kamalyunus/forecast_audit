"""
Training and evaluation logic for forecast adjustment using Linear Agent.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm


class ForecastAdjustmentTrainer:
    """
    Trainer for forecast adjustment using Linear Agent.
    """
    
    def __init__(self, 
                agent,
                environment, 
                output_dir: str = "output",
                num_episodes: int = 500,
                max_steps: int = 14,
                batch_size: int = 32,
                save_every: int = 50,
                optimize_for: str = "both",  # "mape", "bias", or "both"
                logger: Optional[logging.Logger] = None):
        """
        Initialize trainer.
        
        Args:
            agent: LinearAgent
            environment: Forecast environment
            output_dir: Directory for outputs
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            batch_size: Batch size for updates
            save_every: How often to save the model
            optimize_for: Which metric to optimize for ("mape", "bias", or "both")
            logger: Logger instance
        """
        self.agent = agent
        self.env = environment
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.save_every = save_every
        self.optimize_for = optimize_for
        
        # Set up logger
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("ForecastAdjustmentTrainer")
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
        self.mape_improvements = []
        self.bias_improvements = []
        self.training_time = 0
        
        # Adjustment factors
        self.adjustment_factors = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        
    def train(self, verbose: bool = True) -> Dict:
        """
        Train the agent for forecast adjustment.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary of training metrics
        """
        self.logger.info(f"Starting training for {self.num_episodes} episodes")
        start_time = time.time()
        
        # Track best metrics for model saving
        best_score = float('-inf')
        best_mape_improvement = 0.0
        best_bias_improvement = 0.0
        
        # Training loop
        for episode in tqdm(range(1, self.num_episodes + 1), disable=not verbose):
            state = self.env.reset()
            episode_score = 0
            episode_td_errors = []
            
            # Metrics for this episode
            sku_original_mape = {sku: 0 for sku in self.env.skus}
            sku_adjusted_mape = {sku: 0 for sku in self.env.skus}
            sku_original_bias = {sku: 0 for sku in self.env.skus}
            sku_adjusted_bias = {sku: 0 for sku in self.env.skus}
            
            for step in range(self.max_steps):
                adjustments = {}
                
                # Determine adjustments for all SKUs
                for i, sku in enumerate(self.env.skus):
                    # Extract state components
                    forecast_dim, error_dim, feature_dim = self.env.get_feature_dims()
                    
                    # State features for this SKU
                    sku_state = state[i]
                    
                    # Get action from agent
                    action_idx = self.agent.act(sku_state)
                    
                    # Extract forecast from state
                    forecasts = sku_state[:forecast_dim]
                    current_forecast = forecasts[0]  # Current day's forecast
                    
                    # Calculate adjusted forecast based on action
                    adjusted_forecast = self.agent.calculate_adjusted_forecast(action_idx, current_forecast)
                    
                    adjustments[sku] = (action_idx, adjusted_forecast)
                
                # Take step in environment
                next_state, rewards, done, info = self.env.step(adjustments)
                
                # Update episode metrics
                episode_score += sum(rewards.values())
                
                # Update per-SKU metrics
                for sku in self.env.skus:
                    sku_original_mape[sku] = info['original_mape'][sku]
                    sku_adjusted_mape[sku] = info['adjusted_mape'][sku]
                    sku_original_bias[sku] = info['original_bias'][sku]
                    sku_adjusted_bias[sku] = info['adjusted_bias'][sku]
                
                # Update agent for each SKU
                for i, sku in enumerate(self.env.skus):
                    sku_state = state[i]
                    next_sku_state = next_state[i]
                    action_idx, _ = adjustments[sku]
                    
                    # Update agent
                    td_error = self.agent.update(sku_state, action_idx, rewards[sku], next_sku_state, done)
                    episode_td_errors.append(td_error)
                
                # Batch update
                self.agent.batch_update(self.batch_size)
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Calculate MAPE and bias improvements
            avg_original_mape = np.mean(list(sku_original_mape.values()))
            avg_adjusted_mape = np.mean(list(sku_adjusted_mape.values()))
            avg_original_bias = np.mean([abs(b) for b in sku_original_bias.values()])
            avg_adjusted_bias = np.mean([abs(b) for b in sku_adjusted_bias.values()])
            
            mape_improvement = (avg_original_mape - avg_adjusted_mape) / avg_original_mape if avg_original_mape > 0 else 0
            bias_improvement = (avg_original_bias - avg_adjusted_bias) / avg_original_bias if avg_original_bias > 0 else 0
            
            # Store metrics
            self.scores.append(episode_score)
            self.mape_improvements.append(mape_improvement)
            self.bias_improvements.append(bias_improvement)
            
            # Log progress
            if verbose and (episode % 10 == 0 or episode == 1):
                avg_score = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
                avg_mape_imp = np.mean(self.mape_improvements[-100:]) if len(self.mape_improvements) >= 100 else np.mean(self.mape_improvements)
                avg_bias_imp = np.mean(self.bias_improvements[-100:]) if len(self.bias_improvements) >= 100 else np.mean(self.bias_improvements)
                avg_td_error = np.mean(episode_td_errors) if episode_td_errors else 0
                
                self.logger.info(f"Episode {episode}/{self.num_episodes} | "
                              f"Score: {episode_score:.2f} | "
                              f"Avg Score: {avg_score:.2f} | "
                              f"MAPE Imp: {mape_improvement:.4f} | "
                              f"Bias Imp: {bias_improvement:.4f} | "
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
                
            if mape_improvement > best_mape_improvement:
                best_mape_improvement = mape_improvement
                best_mape_model_path = os.path.join(self.model_dir, "best_mape_model.pkl")
                self.agent.save(best_mape_model_path)
                
            if bias_improvement > best_bias_improvement:
                best_bias_improvement = bias_improvement
                best_bias_model_path = os.path.join(self.model_dir, "best_bias_model.pkl")
                self.agent.save(best_bias_model_path)
        
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
            'mape_improvements': self.mape_improvements,
            'bias_improvements': self.bias_improvements,
            'training_time': self.training_time,
            'best_score': best_score,
            'best_mape_improvement': best_mape_improvement,
            'best_bias_improvement': best_bias_improvement,
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
        
        # Plot MAPE improvement
        plt.subplot(2, 2, 2)
        plt.plot(self.mape_improvements)
        plt.title('MAPE Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        
        # Plot bias improvement
        plt.subplot(2, 2, 3)
        plt.plot(self.bias_improvements)
        plt.title('Bias Improvement')
        plt.xlabel('Episode')
        plt.ylabel('Improvement Ratio')
        
        # Plot action distribution
        plt.subplot(2, 2, 4)
        action_counts = self.agent.action_counts
        total_actions = np.sum(action_counts)
        
        if total_actions > 0:
            action_distribution = action_counts / total_actions
            labels = [f"{factor:.1f}x" for factor in self.adjustment_factors]
            plt.bar(labels, action_distribution)
            plt.title('Adjustment Factor Distribution')
            plt.xlabel('Adjustment Factor')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'))
        plt.close()
    
    def evaluate(self, num_episodes: int = 10, verbose: bool = True) -> Dict:
        """
        Evaluate the forecast adjustment agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            verbose: Whether to print progress
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Starting evaluation for {num_episodes} episodes")
        
        # Evaluation metrics
        eval_scores = []
        eval_mape_improvements = []
        eval_bias_improvements = []
        sku_level_metrics = {}
        
        # Initialize SKU-level metrics tracking
        for sku in self.env.skus:
            sku_level_metrics[sku] = {
                "original_mape": [],
                "adjusted_mape": [],
                "original_bias": [],
                "adjusted_bias": [],
                "actions": []
            }
        
        # Evaluation loop
        for episode in tqdm(range(1, num_episodes + 1), disable=not verbose):
            state = self.env.reset()
            episode_score = 0
            
            for step in range(self.max_steps):
                adjustments = {}
                
                # Determine adjustments for all SKUs (no exploration)
                for i, sku in enumerate(self.env.skus):
                    # Extract state components
                    forecast_dim, error_dim, feature_dim = self.env.get_feature_dims()
                    
                    # State features for this SKU
                    sku_state = state[i]
                    
                    # Get action from agent (no exploration)
                    action_idx = self.agent.act(sku_state, explore=False)
                    
                    # Extract forecast from state
                    forecasts = sku_state[:forecast_dim]
                    current_forecast = forecasts[0]  # Current day's forecast
                    
                    # Calculate adjusted forecast based on action
                    adjusted_forecast = self.agent.calculate_adjusted_forecast(action_idx, current_forecast)
                    
                    adjustments[sku] = (action_idx, adjusted_forecast)
                    
                    # Track actions for SKU
                    sku_level_metrics[sku]["actions"].append(action_idx)
                
                # Take step in environment
                next_state, rewards, done, info = self.env.step(adjustments)
                
                # Update episode metrics
                episode_score += sum(rewards.values())
                
                # Update SKU-level metrics
                for sku in self.env.skus:
                    sku_level_metrics[sku]["original_mape"].append(info['original_mape'][sku])
                    sku_level_metrics[sku]["adjusted_mape"].append(info['adjusted_mape'][sku])
                    sku_level_metrics[sku]["original_bias"].append(info['original_bias'][sku])
                    sku_level_metrics[sku]["adjusted_bias"].append(info['adjusted_bias'][sku])
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Calculate episode improvements
            original_mape = np.mean([info['original_mape'][sku] for sku in info['original_mape']])
            adjusted_mape = np.mean([info['adjusted_mape'][sku] for sku in info['adjusted_mape']])
            original_bias = np.mean([abs(info['original_bias'][sku]) for sku in info['original_bias']])
            adjusted_bias = np.mean([abs(info['adjusted_bias'][sku]) for sku in info['adjusted_bias']])
            
            mape_improvement = (original_mape - adjusted_mape) / original_mape if original_mape > 0 else 0
            bias_improvement = (original_bias - adjusted_bias) / original_bias if original_bias > 0 else 0
            
            # Store metrics
            eval_scores.append(episode_score)
            eval_mape_improvements.append(mape_improvement)
            eval_bias_improvements.append(bias_improvement)
            
            if verbose:
                self.logger.info(f"Eval Episode {episode}/{num_episodes} | "
                              f"Score: {episode_score:.2f} | "
                              f"MAPE Imp: {mape_improvement:.4f} | "
                              f"Bias Imp: {bias_improvement:.4f}")
        
        # Calculate aggregate metrics
        avg_score = np.mean(eval_scores)
        avg_mape_improvement = np.mean(eval_mape_improvements)
        avg_bias_improvement = np.mean(eval_bias_improvements)
        
        # Calculate SKU-level summary
        sku_summary = {}
        for sku in self.env.skus:
            orig_mape = np.mean(sku_level_metrics[sku]["original_mape"])
            adj_mape = np.mean(sku_level_metrics[sku]["adjusted_mape"])
            orig_bias = np.mean([abs(b) for b in sku_level_metrics[sku]["original_bias"]])
            adj_bias = np.mean([abs(b) for b in sku_level_metrics[sku]["adjusted_bias"]])
            
            mape_imp = (orig_mape - adj_mape) / orig_mape if orig_mape > 0 else 0
            bias_imp = (orig_bias - adj_bias) / orig_bias if orig_bias > 0 else 0
            
            most_common_action = np.argmax(np.bincount(sku_level_metrics[sku]["actions"]))
            
            sku_summary[sku] = {
                "original_mape": orig_mape,
                "adjusted_mape": adj_mape,
                "mape_improvement": mape_imp,
                "original_bias": orig_bias,
                "adjusted_bias": adj_bias,
                "bias_improvement": bias_imp,
                "common_adjustment": self.adjustment_factors[most_common_action]
            }
        
        # Sort SKUs by improvement
        if self.optimize_for == "mape":
            top_skus = sorted(sku_summary.items(), key=lambda x: x[1]["mape_improvement"], reverse=True)
        elif self.optimize_for == "bias":
            top_skus = sorted(sku_summary.items(), key=lambda x: x[1]["bias_improvement"], reverse=True)
        else:  # "both"
            top_skus = sorted(sku_summary.items(), 
                             key=lambda x: x[1]["mape_improvement"] + x[1]["bias_improvement"], 
                             reverse=True)
        
        # Create summary visualization
        plt.figure(figsize=(15, 10))
        
        # MAPE improvement by SKU (top 20)
        plt.subplot(2, 2, 1)
        if len(top_skus) > 0:
            top_20_skus = [sku for sku, _ in top_skus[:min(20, len(top_skus))]]
            mape_imps = [sku_summary[sku]["mape_improvement"] for sku in top_20_skus]
            plt.bar(range(len(top_20_skus)), mape_imps)
            plt.xticks(range(len(top_20_skus)), top_20_skus, rotation=90)
            plt.title('MAPE Improvement by SKU (Top 20)')
            plt.ylabel('Improvement Ratio')
        else:
            plt.text(0.5, 0.5, "No SKU data available", ha='center', va='center')
            plt.title('MAPE Improvement by SKU')
        
        # Bias improvement by SKU (top 20)
        plt.subplot(2, 2, 2)
        if len(top_skus) > 0:
            bias_imps = [sku_summary[sku]["bias_improvement"] for sku in top_20_skus]
            plt.bar(range(len(top_20_skus)), bias_imps)
            plt.xticks(range(len(top_20_skus)), top_20_skus, rotation=90)
            plt.title('Bias Improvement by SKU (Top 20)')
            plt.ylabel('Improvement Ratio')
        else:
            plt.text(0.5, 0.5, "No SKU data available", ha='center', va='center')
            plt.title('Bias Improvement by SKU')
        
        # Distribution of adjustment factors
        plt.subplot(2, 2, 3)
        adjustment_counts = [0] * len(self.adjustment_factors)
        for sku in self.env.skus:
            for action in sku_level_metrics[sku]["actions"]:
                adjustment_counts[action] += 1
        
        total = sum(adjustment_counts)
        if total > 0:
            adjustment_dist = [count / total for count in adjustment_counts]
            labels = [f"{factor:.1f}x" for factor in self.adjustment_factors]
            plt.bar(labels, adjustment_dist)
            plt.title('Adjustment Factor Distribution')
            plt.ylabel('Frequency')
        
        # Average improvements
        plt.subplot(2, 2, 4)
        plt.bar(['MAPE', 'Bias'], [avg_mape_improvement, avg_bias_improvement])
        plt.title('Average Improvements')
        plt.ylabel('Improvement Ratio')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'evaluation_summary.png'))
        
        self.logger.info(f"Evaluation complete | "
                      f"Avg Score: {avg_score:.2f} | "
                      f"Avg MAPE Imp: {avg_mape_improvement:.4f} | "
                      f"Avg Bias Imp: {avg_bias_improvement:.4f}")
        
        # Return metrics
        metrics = {
            'scores': eval_scores,
            'mape_improvements': eval_mape_improvements,
            'bias_improvements': eval_bias_improvements,
            'avg_score': avg_score,
            'avg_mape_improvement': avg_mape_improvement,
            'avg_bias_improvement': avg_bias_improvement,
            'sku_metrics': sku_summary,
            'top_skus': top_skus
        }
        
        return metrics
    
    def generate_adjusted_forecasts(self, num_days: int = 14) -> pd.DataFrame:
        """
        Generate adjusted forecasts using the trained agent.
        
        Args:
            num_days: Number of days to forecast
            
        Returns:
            DataFrame of adjusted forecasts
        """
        self.logger.info(f"Generating adjusted forecasts for {num_days} days")
        
        # Reset environment
        state = self.env.reset()
        
        # Predictions storage
        forecast_adjustments = []
        
        for day in range(min(num_days, self.max_steps)):
            # Process each SKU
            for i, sku in enumerate(self.env.skus):
                # Extract state components
                forecast_dim, error_dim, feature_dim = self.env.get_feature_dims()
                
                # State features for this SKU
                sku_state = state[i]
                
                # Get action from agent (no exploration)
                action_idx = self.agent.act(sku_state, explore=False)
                
                # Extract forecasts from state
                forecasts = sku_state[:forecast_dim]
                
                # For each forecast day
                for forecast_day in range(forecast_dim):
                    original_forecast = forecasts[forecast_day]
                    
                    # Apply adjustment factor
                    factor = self.adjustment_factors[action_idx]
                    adjusted_forecast = original_forecast * factor
                    
                    # Add to predictions
                    forecast_adjustments.append({
                        'sku_id': sku,
                        'day': day,
                        'forecast_day': forecast_day,
                        'original_forecast': float(original_forecast),
                        'adjustment_factor': float(factor),
                        'adjusted_forecast': float(adjusted_forecast),
                        'action_idx': int(action_idx)
                    })
            
            # Calculate adjustments for environment step
            adjustments = {}
            for i, sku in enumerate(self.env.skus):
                # State features
                sku_state = state[i]
                
                # Get action
                action_idx = self.agent.act(sku_state, explore=False)
                
                # Extract current forecast
                forecast_dim, _, _ = self.env.get_feature_dims()
                current_forecast = sku_state[0]  # First forecast
                
                # Calculate adjusted forecast
                adjusted_forecast = self.agent.calculate_adjusted_forecast(action_idx, current_forecast)
                
                adjustments[sku] = (action_idx, adjusted_forecast)
            
            # Take environment step
            next_state, _, done, _ = self.env.step(adjustments)
            state = next_state
            
            if done:
                break
        
        # Convert to DataFrame
        return pd.DataFrame(forecast_adjustments)