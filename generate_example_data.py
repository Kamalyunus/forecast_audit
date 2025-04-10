"""
Generate example inventory and forecast data files for testing.
"""

import pandas as pd
import numpy as np
import os

def generate_inventory_data(num_skus=50, output_file="example_inventory.csv"):
    """
    Generate example inventory data.
    
    Args:
        num_skus: Number of SKUs to generate
        output_file: Output CSV file path
    """
    # Create SKU IDs
    sku_ids = [f"SKU_{i:04d}" for i in range(num_skus)]
    
    # Generate inventory levels with different patterns
    inventory_data = []
    
    for i, sku_id in enumerate(sku_ids):
        # Create different inventory profiles for variety
        if i % 5 == 0:
            # High inventory SKUs
            inventory = np.random.randint(100, 300)
        elif i % 5 == 1:
            # Medium inventory SKUs
            inventory = np.random.randint(50, 150)
        elif i % 5 == 2:
            # Low inventory SKUs
            inventory = np.random.randint(10, 50)
        elif i % 5 == 3:
            # Very low inventory SKUs
            inventory = np.random.randint(1, 10)
        else:
            # Out of stock SKUs
            inventory = 0
        
        inventory_data.append({
            'sku_id': sku_id,
            'current_inventory': inventory,
            'product_category': f"Category_{(i % 10) + 1}",
            'shelf_life_days': np.random.choice([3, 5, 7, 14]),
            'unit_cost': round(np.random.uniform(1.0, 20.0), 2)
        })
    
    # Create DataFrame
    df = pd.DataFrame(inventory_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Inventory data saved to {output_file}")
    return df

def generate_forecast_data(num_skus=50, forecast_days=14, output_file="example_forecast.csv"):
    """
    Generate example forecast data.
    
    Args:
        num_skus: Number of SKUs to generate
        forecast_days: Number of days to forecast
        output_file: Output CSV file path
    """
    # Create SKU IDs
    sku_ids = [f"SKU_{i:04d}" for i in range(num_skus)]
    
    # Generate forecasts with different patterns
    forecast_data = []
    
    for i, sku_id in enumerate(sku_ids):
        # Create base demand with seasonal patterns
        if i % 4 == 0:
            # High demand SKUs with weekly pattern
            base_demand = np.random.randint(50, 150)
            weekly_pattern = np.array([0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.2])  # Weekend peaks
        elif i % 4 == 1:
            # Medium demand SKUs with weekly pattern
            base_demand = np.random.randint(20, 60)
            weekly_pattern = np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.9])  # Weekday peaks
        elif i % 4 == 2:
            # Low demand SKUs with weekly pattern
            base_demand = np.random.randint(5, 20)
            weekly_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Flat pattern
        else:
            # Very low demand SKUs with weekly pattern
            base_demand = np.random.randint(1, 5)
            weekly_pattern = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 1.0])  # Strong weekend peaks
        
        # Create row with SKU info
        row = {'sku_id': sku_id}
        
        # Generate ML forecasts
        for day in range(1, forecast_days + 1):
            # Add weekly seasonality
            day_of_week = (day - 1) % 7
            seasonality = weekly_pattern[day_of_week]
            
            # Add trend component
            trend = 1.0 + 0.01 * day if i % 3 == 0 else 1.0
            
            # Generate forecast with some noise
            ml_forecast = base_demand * seasonality * trend * np.random.normal(1.0, 0.05)
            row[f'ml_day_{day}'] = max(0, round(ml_forecast))
            
            # Generate ARIMA forecast (similar but slightly different)
            arima_forecast = base_demand * seasonality * trend * np.random.normal(1.0, 0.08)
            row[f'arima_day_{day}'] = max(0, round(arima_forecast))
        
        # Add forecast accuracy metrics
        row['ml_mape_7d'] = np.random.uniform(0.15, 0.35)
        row['ml_mape_30d'] = np.random.uniform(0.20, 0.40)
        row['ml_bias_7d'] = np.random.uniform(-0.2, 0.2)
        row['ml_bias_30d'] = np.random.uniform(-0.15, 0.15)
        
        row['arima_mape_7d'] = np.random.uniform(0.20, 0.40)
        row['arima_mape_30d'] = np.random.uniform(0.25, 0.45)
        row['arima_bias_7d'] = np.random.uniform(-0.25, 0.25)
        row['arima_bias_30d'] = np.random.uniform(-0.20, 0.20)
        
        forecast_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(forecast_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Forecast data saved to {output_file}")
    return df

def generate_price_data(num_skus=50, num_days=14, output_file="example_price.csv"):
    """
    Generate example price data.
    
    Args:
        num_skus: Number of SKUs to generate
        num_days: Number of days to generate prices for
        output_file: Output CSV file path
    """
    # Create price data
    price_data = []
    for i in range(num_skus):
        sku_id = f"SKU_{i:04d}"
        # Base price for the SKU
        base_price = round(np.random.uniform(5.0, 50.0), 2)
        
        for day in range(num_days):
            # Occasional price changes
            price_adjustment = 0 if np.random.random() > 0.1 else np.random.choice([-0.1, -0.05, 0.05, 0.1])
            price = round(base_price * (1 + price_adjustment), 2)
            
            price_data.append({
                'sku_id': sku_id,
                'price_date': f"2023-01-{day+1:02d}",
                'price': price
            })
    
    # Create DataFrame
    df = pd.DataFrame(price_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Price data saved to {output_file}")
    return df

def generate_all_example_files(num_skus=50):
    """Generate all example data files."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate inventory data
    inventory_df = generate_inventory_data(num_skus=num_skus)
    
    # Generate forecast data
    forecast_df = generate_forecast_data(num_skus=num_skus)
    
    # Generate price data
    price_df = generate_price_data(num_skus=num_skus)
    
    print("\nExample files created successfully.")
    print("To run the example, use the command:")
    print("python main.py --mode train --inventory-file example_inventory.csv --forecast-file example_forecast.csv --price-file example_price.csv --episodes 100 --output-dir example_output")
    
    # Print data samples
    print("\nInventory data sample:")
    print(inventory_df.head().to_string())
    
    print("\nForecast data sample (first few columns):")
    forecast_sample_cols = ['sku_id'] + [col for col in forecast_df.columns if col.startswith('ml_day_')][:3]
    print(forecast_df[forecast_sample_cols].head().to_string())

if __name__ == "__main__":
    generate_all_example_files()