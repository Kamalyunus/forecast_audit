"""
Environment for forecast adjustment with reinforcement learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging


class ForecastEnvironment:
    """
    Environment for training RL agents to adjust forecasts.
    """
    
    def __init__(self, 
                forecast_data: pd.DataFrame,
                historical_data: Optional[pd.DataFrame] = None,
                validation_length: int = 30,
                forecast_horizon: int = 7,
                optimize_for: str = "both",  # "mape", "bias", or "both"
                logger: Optional[logging.Logger] = None):
        """
        Initialize the forecast adjustment environment.
        
        Args:
            forecast_data: DataFrame with forecast data (ml_day_X columns)
            historical_data: Optional DataFrame with actual historical values for validation
            validation_length: Number of days to use for validation
            forecast_horizon: Number of days in the forecast horizon
            optimize_for: Which metric to optimize for ("mape", "bias", or "both")
            logger: Optional logger instance
        """
        self.forecast_data = forecast_data
        self.historical_data = historical_data
        self.validation_length = validation_length
        self.forecast_horizon = forecast_horizon
        self.optimize_for = optimize_for
        
        # Set up logger
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("ForecastEnvironment")
        else:
            self.logger = logger
            
        # Extract SKUs
        self.skus = self.forecast_data['sku_id'].unique().tolist()
        self.logger.info(f"Environment initialized with {len(self.skus)} SKUs")
        
        # Extract ML forecast columns
        self.ml_cols = [col for col in self.forecast_data.columns if col.startswith('ml_day_')]
        
        # Extract accuracy metrics
        self.accuracy_cols = [
            'ml_mape_7d', 'ml_mape_30d', 'ml_bias_7d', 'ml_bias_30d'
        ]
        
        # Create validation data if historical data is provided
        self.has_validation = False
        if historical_data is not None:
            self._setup_validation_data()
            self.has_validation = True
            
        # Current state of the environment
        self.current_step = 0
        self.current_state = None
        self.done = False
        
        # History tracking
        self.adjustment_history = []
    
    def _setup_validation_data(self):
        """
        Set up validation data from historical data.
        """
        self.validation_data = {}
        
        # Organize historical data by SKU and date
        if 'sale_date' in self.historical_data.columns and 'quantity' in self.historical_data.columns:
            for sku in self.skus:
                sku_history = self.historical_data[self.historical_data['sku_id'] == sku].copy()
                if not sku_history.empty:
                    try:
                        # Try to parse dates with flexible format
                        sku_history['sale_date'] = pd.to_datetime(sku_history['sale_date'], format='mixed')
                        sku_history = sku_history.sort_values('sale_date')
                        
                        # Use the most recent data for validation
                        if len(sku_history) >= self.validation_length:
                            self.validation_data[sku] = sku_history.tail(self.validation_length)['quantity'].values
                        else:
                            # Pad with zeros if not enough data
                            actual_values = sku_history['quantity'].values
                            self.validation_data[sku] = np.pad(
                                actual_values, 
                                (0, self.validation_length - len(actual_values)),
                                'constant'
                            )
                    except Exception as e:
                        self.logger.warning(f"Error processing dates for SKU {sku}: {str(e)}")
                        # If date processing fails, sort by index as fallback
                        if len(sku_history) >= self.validation_length:
                            self.validation_data[sku] = sku_history.tail(self.validation_length)['quantity'].values
                        else:
                            actual_values = sku_history['quantity'].values
                            self.validation_data[sku] = np.pad(
                                actual_values, 
                                (0, self.validation_length - len(actual_values)),
                                'constant'
                            )
        
        # If not enough validation data, create synthetic data
        missing_skus = set(self.skus) - set(self.validation_data.keys())
        if missing_skus:
            self.logger.warning(f"Creating synthetic validation data for {len(missing_skus)} SKUs")
            
            for sku in missing_skus:
                # Get forecast data for this SKU
                sku_forecast = self.forecast_data[self.forecast_data['sku_id'] == sku]
                
                if not sku_forecast.empty:
                    # Extract ML forecasts for the first few days
                    ml_forecasts = []
                    for col in self.ml_cols[:self.validation_length]:
                        if col in sku_forecast.columns:
                            ml_forecasts.append(sku_forecast[col].iloc[0])
                        else:
                            ml_forecasts.append(0)
                    
                    # Create synthetic actuals by adding noise to forecasts
                    if 'ml_mape_7d' in sku_forecast.columns:
                        mape = sku_forecast['ml_mape_7d'].iloc[0]
                    else:
                        mape = 0.2  # Default 20% MAPE
                        
                    if 'ml_bias_7d' in sku_forecast.columns:
                        bias = sku_forecast['ml_bias_7d'].iloc[0]
                    else:
                        bias = 0.0  # Default no bias
                    
                    # Apply bias and random error based on MAPE
                    synthetic_actuals = []
                    for forecast in ml_forecasts[:self.validation_length]:
                        # Apply bias
                        biased_forecast = forecast * (1 + bias)
                        
                        # Apply random error based on MAPE
                        error_range = biased_forecast * mape
                        random_error = np.random.uniform(-error_range, error_range)
                        
                        # Final synthetic actual
                        synthetic_actual = max(0, biased_forecast + random_error)
                        synthetic_actuals.append(synthetic_actual)
                    
                    self.validation_data[sku] = np.array(synthetic_actuals)
        
        self.logger.info(f"Validation data prepared for {len(self.validation_data)} SKUs")
    
    def get_feature_dims(self) -> Tuple[int, int, int]:
        """
        Get dimensions of the state features.
        
        Returns:
            Tuple of (forecast_dim, error_metrics_dim, total_feature_dim)
        """
        forecast_dim = min(len(self.ml_cols), self.forecast_horizon)
        error_metrics_dim = len(self.accuracy_cols)
        
        # Total dimensions: forecasts + error metrics 
        total_dim = forecast_dim + error_metrics_dim
        
        return forecast_dim, error_metrics_dim, total_dim
    
    def _get_sku_state(self, sku: str, step: int) -> np.ndarray:
        """
        Get state representation for a specific SKU.
        
        Args:
            sku: SKU identifier
            step: Current step in the episode
            
        Returns:
            State features for the SKU
        """
        sku_forecast = self.forecast_data[self.forecast_data['sku_id'] == sku]
        
        if sku_forecast.empty:
            # Default state if SKU not found
            forecast_dim, error_dim, _ = self.get_feature_dims()
            return np.zeros(forecast_dim + error_dim)
        
        # 1. Extract forecasts for the next N days
        forecasts = []
        for i in range(self.forecast_horizon):
            day_idx = step + i
            if day_idx < len(self.ml_cols):
                ml_col = self.ml_cols[day_idx]
                
                if ml_col in sku_forecast.columns:
                    ml_value = sku_forecast[ml_col].iloc[0]
                    forecasts.append(ml_value)
                else:
                    forecasts.append(0.0)
            else:
                forecasts.append(0.0)
        
        # 2. Extract forecast accuracy metrics
        error_metrics = []
        for col in self.accuracy_cols:
            if col in sku_forecast.columns:
                error_metrics.append(sku_forecast[col].iloc[0])
            else:
                error_metrics.append(0.0)
        
        # Combine features
        state = np.concatenate([forecasts, error_metrics]).astype(np.float32)
        
        return state
    
    def reset(self) -> List[np.ndarray]:
        """
        Reset the environment to the initial state.
        
        Returns:
            Initial state for all SKUs
        """
        self.current_step = 0
        self.done = False
        self.adjustment_history = []
        
        # Get initial state for all SKUs
        self.current_state = [self._get_sku_state(sku, self.current_step) for sku in self.skus]
        
        return self.current_state
    
    def step(self, actions: Dict[str, Tuple[int, float]]) -> Tuple[List[np.ndarray], Dict[str, float], bool, Dict]:
        """
        Take a step in the environment by applying forecast adjustments.
        
        Args:
            actions: Dictionary mapping SKU to (action_idx, adjusted_forecast)
            
        Returns:
            Tuple of (next_state, rewards, done, info)
        """
        rewards = {}
        info = {
            'original_mape': {},
            'adjusted_mape': {},
            'original_bias': {},
            'adjusted_bias': {}
        }
        
        # Process actions for each SKU
        for i, sku in enumerate(self.skus):
            if sku not in actions:
                rewards[sku] = 0.0
                continue
                
            action_idx, adjusted_forecast = actions[sku]
            
            # Get original forecast
            sku_forecast = self.forecast_data[self.forecast_data['sku_id'] == sku]
            
            if not sku_forecast.empty and self.current_step < len(self.ml_cols):
                ml_col = self.ml_cols[self.current_step]
                
                if ml_col in sku_forecast.columns:
                    original_forecast = sku_forecast[ml_col].iloc[0]
                else:
                    original_forecast = 0.0
            else:
                original_forecast = 0.0
            
            # Calculate reward based on validation data (if available)
            if self.has_validation and sku in self.validation_data and self.current_step < len(self.validation_data[sku]):
                actual = self.validation_data[sku][self.current_step]
                
                # Calculate original error metrics
                if actual > 0:
                    original_mape = abs(original_forecast - actual) / actual
                    original_bias = (original_forecast - actual) / actual
                else:
                    original_mape = 1.0 if original_forecast > 0 else 0.0
                    original_bias = 1.0 if original_forecast > 0 else 0.0
                
                # Calculate adjusted error metrics
                if actual > 0:
                    adjusted_mape = abs(adjusted_forecast - actual) / actual
                    adjusted_bias = (adjusted_forecast - actual) / actual
                else:
                    adjusted_mape = 1.0 if adjusted_forecast > 0 else 0.0
                    adjusted_bias = 1.0 if adjusted_forecast > 0 else 0.0
                
                # Calculate reward based on improvement
                mape_improvement = original_mape - adjusted_mape
                bias_improvement = abs(original_bias) - abs(adjusted_bias)
                
                # Reward based on optimization target
                if self.optimize_for == "mape":
                    reward = mape_improvement * 10  # Scale for better learning
                elif self.optimize_for == "bias":
                    reward = bias_improvement * 10
                else:  # "both"
                    reward = (mape_improvement + bias_improvement) * 5
                
                # Store metrics in info
                info['original_mape'][sku] = float(original_mape)
                info['adjusted_mape'][sku] = float(adjusted_mape)
                info['original_bias'][sku] = float(original_bias)
                info['adjusted_bias'][sku] = float(adjusted_bias)
            else:
                # No validation data, use forecast accuracy metrics as proxy
                sku_forecast = self.forecast_data[self.forecast_data['sku_id'] == sku]
                
                if not sku_forecast.empty and 'ml_mape_7d' in sku_forecast.columns:
                    mape = sku_forecast['ml_mape_7d'].iloc[0]
                    bias = sku_forecast['ml_bias_7d'].iloc[0] if 'ml_bias_7d' in sku_forecast.columns else 0.0
                    
                    # Use historical accuracy to estimate current error
                    est_original_mape = mape
                    est_original_bias = bias
                    
                    # Estimate adjustment impact based on action
                    adjustment_factor = adjusted_forecast / original_forecast if original_forecast > 0 else 1.0
                    
                    # Simple heuristic: if historical bias is positive (overforecast), 
                    # reducing forecast (factor < 1) might help
                    if bias > 0 and adjustment_factor < 1.0:
                        est_bias_improvement = bias * (1.0 - adjustment_factor)
                    # If historical bias is negative (underforecast), 
                    # increasing forecast (factor > 1) might help
                    elif bias < 0 and adjustment_factor > 1.0:
                        est_bias_improvement = abs(bias) * (adjustment_factor - 1.0)
                    else:
                        # Going in the wrong direction
                        est_bias_improvement = -abs(bias) * abs(adjustment_factor - 1.0)
                    
                    # For MAPE, smaller adjustments tend to be safer
                    est_mape_factor = abs(adjustment_factor - 1.0)
                    est_mape_improvement = -est_mape_factor * mape
                    
                    # Create a heuristic reward
                    if self.optimize_for == "mape":
                        reward = est_mape_improvement * 5
                    elif self.optimize_for == "bias":
                        reward = est_bias_improvement * 5
                    else:  # "both"
                        reward = (est_mape_improvement + est_bias_improvement) * 2.5
                    
                    # Populate info with estimates
                    info['original_mape'][sku] = float(est_original_mape)
                    info['adjusted_mape'][sku] = float(est_original_mape + est_mape_improvement)
                    info['original_bias'][sku] = float(est_original_bias)
                    info['adjusted_bias'][sku] = float(est_original_bias - est_bias_improvement)
                else:
                    # No forecast accuracy data
                    reward = 0.0
                    info['original_mape'][sku] = 0.0
                    info['adjusted_mape'][sku] = 0.0
                    info['original_bias'][sku] = 0.0
                    info['adjusted_bias'][sku] = 0.0
            
            rewards[sku] = reward
            
            # Track adjustment
            self.adjustment_history.append({
                'step': self.current_step,
                'sku': sku,
                'original_forecast': original_forecast,
                'adjusted_forecast': adjusted_forecast,
                'action_idx': action_idx,
                'reward': reward
            })
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= min(len(self.ml_cols), self.validation_length):
            self.done = True
            
        # Get next state
        next_state = [self._get_sku_state(sku, self.current_step) for sku in self.skus]
        self.current_state = next_state
        
        return next_state, rewards, self.done, info
    
    def get_adjustment_history(self) -> pd.DataFrame:
        """
        Get the history of forecast adjustments.
        
        Returns:
            DataFrame of adjustment history
        """
        if not self.adjustment_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.adjustment_history)
    
    def calculate_accuracy_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate overall forecast accuracy metrics for original and adjusted forecasts.
        
        Returns:
            Dictionary of metrics
        """
        if not self.adjustment_history:
            return {}
        
        history_df = self.get_adjustment_history()
        metrics = {}
        
        # Calculate metrics per SKU
        for sku in self.skus:
            sku_history = history_df[history_df['sku'] == sku]
            if sku_history.empty:
                continue
            
            # Calculate MAPE and Bias if validation data exists
            if self.has_validation and sku in self.validation_data:
                sku_actuals = self.validation_data[sku][:len(sku_history)]
                
                # Filter out zero actuals for MAPE calculation
                non_zero_idx = np.where(sku_actuals > 0)[0]
                
                if len(non_zero_idx) > 0:
                    # Original forecasts
                    orig_forecasts = sku_history['original_forecast'].values
                    orig_non_zero = orig_forecasts[non_zero_idx]
                    actuals_non_zero = sku_actuals[non_zero_idx]
                    
                    orig_mape = np.mean(np.abs(orig_non_zero - actuals_non_zero) / actuals_non_zero)
                    orig_bias = np.mean((orig_non_zero - actuals_non_zero) / actuals_non_zero)
                    
                    # Adjusted forecasts
                    adj_forecasts = sku_history['adjusted_forecast'].values
                    adj_non_zero = adj_forecasts[non_zero_idx]
                    
                    adj_mape = np.mean(np.abs(adj_non_zero - actuals_non_zero) / actuals_non_zero)
                    adj_bias = np.mean((adj_non_zero - actuals_non_zero) / actuals_non_zero)
                    
                    # Calculate improvements
                    mape_improvement = (orig_mape - adj_mape) / orig_mape if orig_mape > 0 else 0
                    bias_improvement = (abs(orig_bias) - abs(adj_bias)) / abs(orig_bias) if orig_bias != 0 else 0
                    
                    metrics[sku] = {
                        'original_mape': orig_mape,
                        'adjusted_mape': adj_mape,
                        'mape_improvement': mape_improvement,
                        'original_bias': orig_bias,
                        'adjusted_bias': adj_bias,
                        'bias_improvement': bias_improvement,
                        'sample_size': len(non_zero_idx)
                    }
        
        return metrics