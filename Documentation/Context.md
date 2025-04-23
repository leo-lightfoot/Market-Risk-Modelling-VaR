# Market Risk Modeling

## Project Overview
This project focuses on pricing and valuing a portfolio of different positions using a systematic approach for generating synthetic price data and evaluating model performance.

## Key Components and Workflow

### 1. Portfolio Creation and Pricing Model Structure
- **Data Import**: Load market data (prices, indices, rates) from CSV files in Raw_Data folder
- **Return Calculation**: Compute log returns for both target assets and risk factors
- **Factor Modeling**: Use regression to model relationships between assets and market factors
- **Synthetic NAV Generation**: Create synthetic prices based on factor model predictions
- **Performance Analysis**: Compare synthetic vs. actual values using statistical metrics

### 2. Market Data and Risk Factors
- Use real market data from the Raw_Data folder
- Key risk factors may include:
  - Currency indices (e.g., DXY)
  - Commodity spot prices (e.g., XAU)
  - Interest rates (e.g., 10Y Treasury yields)
- Create interaction terms between factors to capture non-linear relationships

### 3. Model Implementation
- Data preprocessing: cleaning, date formatting, return calculation
- Model training: Ridge regression with cross-validation for optimal regularization
- Output generation: synthetic NAVs and returns saved to NAV_returns_Data folder
- Visualization: comparison plots saved to respective asset folders

### 4. Performance Metrics
- R-squared: Measures explanatory power of the model
- RMSE: Quantifies prediction error
- Correlation: Measures similarity between actual and synthetic values
- Visual comparison: Time series plots of actual vs. synthetic prices
