# Market Risk Modeling Framework

## Overview
This repository contains a framework for pricing and valuing portfolio positions using factor-based modeling with real market data. The system generates synthetic NAV (Net Asset Value) values based on regression models that capture relationships between assets and their risk factors.

## Project Structure
- `Raw_Data/`: Contains raw market data (prices, indices, rates)
- `NAV_returns_Data/`: Stores output CSV files with synthetic NAVs and returns
- `Gold/`: Asset-specific folder for Gold modeling with visualization outputs
- `pricing_model_template.py`: Reusable template for creating asset-specific pricing models
- `Context.md`: Project documentation with workflow and implementation details

## Pricing Model Approach
The pricing model follows these key steps:

1. **Data Preparation**
   - Load market data from CSV files
   - Process dates and clean column names
   - Set up target asset and risk factors

2. **Return Calculation**
   - Compute log returns for the target asset
   - Compute log returns for risk factors
   - Create interaction terms to capture non-linear relationships

3. **Model Training and Evaluation**
   - Use Ridge regression with cross-validation
   - Train on a portion of data (80%) and test on the remainder
   - Evaluate using R², RMSE, and other metrics

4. **Synthetic NAV Generation**
   - Predict returns using the trained model
   - Convert returns to prices using cumulative sum technique
   - Compare synthetic vs. actual values

5. **Output Generation**
   - Save synthetic NAVs and returns to CSV files
   - Create visualization comparing actual vs. synthetic prices
   - Calculate performance statistics

## Usage Example
The package provides a modular framework that can be adapted for different assets:

```python
# Import the pricing model template
from pricing_model_template import run_pricing_model

# Define asset parameters
asset_name = "GOLD"  
raw_data_path = "Raw_Data/combined_data_gold.csv"
target_column = "GLD US Equity (USD)"
factor_columns = ["XAU Curncy", "DXY Curncy", "USGG10Y Index"]

# Run the model
output_df, stats = run_pricing_model(
    asset_name, 
    raw_data_path, 
    target_column, 
    factor_columns
)

# Print performance metrics
print(f"Model performance: R² = {stats['correlation']:.4f}")
```

## Model Performance
The model's effectiveness is evaluated using several metrics:
- **R-squared**: Measures how well the factors explain asset returns
- **RMSE**: Root Mean Square Error between synthetic and actual values
- **Correlation**: Between synthetic and actual NAVs
- **Visual comparison**: Time series plots showing model accuracy

## Extending the Framework
To add a new asset to the framework:
1. Add market data to the `Raw_Data/` folder
2. Create an asset-specific implementation using the template
3. Define appropriate risk factors for the asset
4. Run the model to generate synthetic data
5. Evaluate performance using provided metrics 