# Forecast Adjustment System

This repository contains a cleaned and simplified implementation of a machine learning system for adjusting forecasts using reinforcement learning. The system learns to apply correction factors to ML-generated forecasts to improve accuracy metrics like MAPE (Mean Absolute Percentage Error) and bias.

## Overview

The system uses a Q-learning approach with linear function approximation to learn optimal adjustment factors for each SKU (Stock Keeping Unit) based on forecast patterns and historical accuracy. It can optimize for MAPE reduction, bias reduction, or both.

## Key Components

1. **Forecast Environment (`forecast_environment.py`)**: 
   - Simulates the forecast adjustment process
   - Calculates rewards based on improvements in accuracy metrics
   - Handles validation using historical or synthetic data

2. **Linear Agent (`linear_agent.py`)**: 
   - Implements Q-learning with linear function approximation
   - Uses epsilon-greedy exploration to discover optimal policies
   - Applies learned adjustment factors to forecasts

3. **Trainer (`trainer.py`)**: 
   - Manages the training, evaluation, and forecast generation process
   - Tracks metrics and visualizes progress
   - Saves model checkpoints

4. **Data Generation (`generate_example_data.py`)**: 
   - Creates synthetic forecast data for testing
   - Generates realistic patterns with weekly seasonality

5. **Main Script (`main.py`)**: 
   - Command-line interface for training, evaluation, and generating adjustments
   - Handles data loading and parameter configuration

6. **Example Script (`example.py`)**: 
   - Provides end-to-end examples of the system in action
   - Demonstrates how to optimize for different metrics

## Adjustment Factors

The system learns to select from a set of adjustment factors for each forecast:
- 0.7x (reduce forecast by 30%)
- 0.8x (reduce forecast by 20%)
- 0.9x (reduce forecast by 10%)
- 1.0x (no adjustment)
- 1.1x (increase forecast by 10%)
- 1.2x (increase forecast by 20%)
- 1.3x (increase forecast by 30%)

## Usage

### Generate Example Data

```python
python generate_example_data.py
```

### Train a Model

```
python main.py --mode train --forecast-file example_forecast.csv --episodes 100 --output-dir example_output
```

### Evaluate a Model

```
python main.py --mode evaluate --forecast-file example_forecast.csv --model-path example_output/models/final_model.pkl --output-dir example_output
```

### Generate Adjusted Forecasts

```
python main.py --mode adjust --forecast-file example_forecast.csv --model-path example_output/models/final_model.pkl --output-dir example_output
```

### Run Example

```
python example.py
```

## Input Data Format

The forecast data should be a CSV file with the following columns:
- `sku_id`: Identifier for each product/SKU
- `ml_day_1`, `ml_day_2`, etc.: ML forecasts for each day
- `ml_mape_7d`, `ml_mape_30d`: Historical MAPE metrics (optional)
- `ml_bias_7d`, `ml_bias_30d`: Historical bias metrics (optional)

## Output

The system generates:
1. Trained models (saved as pickle files)
2. Visualization of training progress
3. Evaluation metrics showing improvement over original forecasts
4. Adjusted forecasts in CSV format

## Customization

The system can be customized through command-line arguments:
- Optimization target (MAPE, bias, or both)
- Learning parameters (learning rate, discount factor, exploration)
- Training parameters (episodes, batch size)
- Forecast parameters (horizon, validation length)

## Dependencies

- NumPy
- Pandas
- Matplotlib
- tqdm