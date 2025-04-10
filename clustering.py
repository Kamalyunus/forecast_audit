"""
SKU clustering utility for large-scale inventory management.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
import logging


def cluster_skus(forecast_data: pd.DataFrame, 
                inventory_data: pd.DataFrame, 
                historical_data: Optional[pd.DataFrame] = None, 
                num_clusters: int = 20,
                logger: Optional[logging.Logger] = None) -> Dict[str, int]:
    """
    Cluster SKUs based on their features.
    
    Args:
        forecast_data: DataFrame with forecast data
        inventory_data: DataFrame with inventory data
        historical_data: Optional DataFrame with historical sales
        num_clusters: Number of clusters to create
        logger: Optional logger instance
        
    Returns:
        Dictionary mapping SKU ID to cluster ID
    """
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("Clustering")
    
    logger.info(f"Clustering {len(forecast_data['sku_id'].unique())} SKUs into {num_clusters} clusters")
    
    # Extract features for clustering
    features = []
    sku_ids = []
    
    for sku in forecast_data['sku_id'].unique():
        sku_ids.append(sku)
        
        sku_forecasts = forecast_data[forecast_data['sku_id'] == sku]
        sku_inventory = inventory_data[inventory_data['sku_id'] == sku]
        
        # Features for clustering
        sku_features = []
        
        # 1. Forecast-related features
        # 1.1 Forecast mean and variance
        ml_cols = [col for col in sku_forecasts.columns if col.startswith('ml_day_')]
        arima_cols = [col for col in sku_forecasts.columns if col.startswith('arima_day_')]
        
        if ml_cols:
            ml_values = sku_forecasts[ml_cols].iloc[0].values
            ml_mean = np.mean(ml_values)
            ml_std = np.std(ml_values)
            sku_features.extend([ml_mean, ml_std])
            
            # Weekly pattern (if we have at least 7 days of forecast)
            if len(ml_cols) >= 7:
                weekly_pattern = ml_values[:7] / (ml_mean + 1e-10)  # Avoid division by zero
                sku_features.extend(weekly_pattern)
        
        if arima_cols:
            arima_values = sku_forecasts[arima_cols].iloc[0].values
            arima_mean = np.mean(arima_values)
            arima_std = np.std(arima_values)
            sku_features.extend([arima_mean, arima_std])
        
        # 1.2 Forecast accuracy measures
        accuracy_features = []
        for col in ['ml_mape_7d', 'ml_mape_30d', 'ml_bias_7d', 'ml_bias_30d',
                   'arima_mape_7d', 'arima_mape_30d', 'arima_bias_7d', 'arima_bias_30d']:
            if col in sku_forecasts.columns:
                accuracy_features.append(sku_forecasts[col].iloc[0])
            else:
                accuracy_features.append(0.0)  # Default value if not available
        
        sku_features.extend(accuracy_features)
        
        # 2. Inventory-related features
        if not sku_inventory.empty and 'current_inventory' in sku_inventory.columns:
            current_inventory = sku_inventory['current_inventory'].iloc[0]
        else:
            current_inventory = 0
            
        # Inventory relative to forecast
        if ml_mean > 0:
            inventory_coverage = current_inventory / (ml_mean + 1e-10)
        else:
            inventory_coverage = 0
            
        sku_features.extend([current_inventory, inventory_coverage])
        
        # 3. Historical data features (if available)
        if historical_data is not None:
            sku_history = historical_data[historical_data['sku_id'] == sku]
            
            if not sku_history.empty:
                # Average historical sales and variability
                avg_sales = sku_history['quantity'].mean()
                sales_std = sku_history['quantity'].std()
                
                # Weekly pattern from historical data (if dates available)
                if 'sale_date' in sku_history.columns:
                    sku_history['day_of_week'] = pd.to_datetime(sku_history['sale_date']).dt.dayofweek
                    dow_pattern = sku_history.groupby('day_of_week')['quantity'].mean()
                    
                    # Fill missing days
                    dow_pattern = [dow_pattern.get(i, avg_sales) for i in range(7)]
                    
                    # Normalize
                    if avg_sales > 0:
                        dow_pattern = [x / (avg_sales + 1e-10) for x in dow_pattern]
                    else:
                        dow_pattern = [1.0] * 7
                    
                    sku_features.extend(dow_pattern)
                
                sku_features.extend([avg_sales, sales_std])
            else:
                # Default values if no history
                sku_features.extend([0.0, 0.0])
                if 'sale_date' in historical_data.columns:
                    sku_features.extend([1.0] * 7)  # Default day of week pattern
        
        features.append(sku_features)
    
    # Ensure all feature vectors have the same length
    max_len = max(len(f) for f in features)
    features_padded = [f + [0.0] * (max_len - len(f)) for f in features]
    
    # Convert to numpy array
    features_array = np.array(features_padded)
    
    # Check for NaN values and replace with 0
    features_array = np.nan_to_num(features_array, nan=0.0)
    
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_array)
    
    # Perform K-means clustering
    logger.info("Running K-means clustering...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(scaled_features)
    
    # Create mapping
    cluster_mapping = {sku: int(cluster_id) for sku, cluster_id in zip(sku_ids, cluster_ids)}
    
    # Log cluster sizes
    cluster_counts = np.bincount(cluster_ids)
    for i, count in enumerate(cluster_counts):
        logger.info(f"Cluster {i}: {count} SKUs")
    
    logger.info("Clustering complete")
    
    return cluster_mapping


def create_similarity_based_mapping(cluster_mapping: Dict[str, int], 
                                  forecast_data: pd.DataFrame,
                                  inventory_data: pd.DataFrame,
                                  new_skus: List[str]) -> Dict[str, int]:
    """
    Create cluster mapping for new SKUs based on similarity to existing clustered SKUs.
    
    Args:
        cluster_mapping: Existing mapping of SKU ID to cluster ID
        forecast_data: DataFrame with forecast data (including new SKUs)
        inventory_data: DataFrame with inventory data
        new_skus: List of new SKU IDs to assign clusters
        
    Returns:
        Updated mapping including new SKUs
    """
    # Extract existing SKUs and their clusters
    existing_skus = list(cluster_mapping.keys())
    existing_clusters = list(cluster_mapping.values())
    
    # Create feature vectors for existing SKUs
    existing_features = []
    for sku in existing_skus:
        sku_forecasts = forecast_data[forecast_data['sku_id'] == sku]
        if sku_forecasts.empty:
            continue
            
        # Simple features - just average forecast and current inventory
        ml_cols = [col for col in sku_forecasts.columns if col.startswith('ml_day_')]
        avg_forecast = sku_forecasts[ml_cols].iloc[0].mean() if ml_cols else 0
        
        sku_inventory = inventory_data[inventory_data['sku_id'] == sku]
        current_inventory = sku_inventory['current_inventory'].iloc[0] if not sku_inventory.empty else 0
        
        existing_features.append([avg_forecast, current_inventory])
    
    # Create feature vectors for new SKUs
    new_features = []
    for sku in new_skus:
        sku_forecasts = forecast_data[forecast_data['sku_id'] == sku]
        if sku_forecasts.empty:
            # If no forecast data, assign to a default cluster (0)
            cluster_mapping[sku] = 0
            continue
            
        # Simple features - just average forecast and current inventory
        ml_cols = [col for col in sku_forecasts.columns if col.startswith('ml_day_')]
        avg_forecast = sku_forecasts[ml_cols].iloc[0].mean() if ml_cols else 0
        
        sku_inventory = inventory_data[inventory_data['sku_id'] == sku]
        current_inventory = sku_inventory['current_inventory'].iloc[0] if not sku_inventory.empty else 0
        
        new_features.append([avg_forecast, current_inventory, sku])
    
    # Normalize features
    all_features = np.array(existing_features)
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Assign clusters to new SKUs based on nearest neighbors
    for features in new_features:
        avg_forecast, current_inventory, sku = features
        scaled_features = scaler.transform([[avg_forecast, current_inventory]])[0]
        
        # Find closest existing SKU
        min_distance = float('inf')
        best_cluster = 0
        
        for i, existing_feature in enumerate(scaler.transform(all_features)):
            distance = np.sqrt(np.sum((scaled_features - existing_feature) ** 2))
            if distance < min_distance:
                min_distance = distance
                best_cluster = existing_clusters[i]
        
        # Assign cluster
        cluster_mapping[sku] = best_cluster
    
    return cluster_mapping