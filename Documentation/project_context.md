# Market Risk Modeling - VaR Calculation Project

## Project Overview
This project focuses on implementing and comparing different Value at Risk (VaR) calculation methodologies for a mixed portfolio of financial instruments.

## Key Requirements

### 1. Portfolio Creation
- Create a portfolio using real or hypothetical positions
- Implement appropriate pricing functions for position valuation
- Portfolio should represent realistic market behavior

### 2. Market Data
- Options:
  - Use real market data
  - Generate synthetic market data using time series models
- Minimum timeframe: 5 years of historical data
- Data should capture various market conditions

### 3. VaR Calculation Methods
Implementation of multiple approaches:
- Delta-Normal method (parametric)
- Historical Simulation
- Monte-Carlo Simulation
- GARCH models
- EVT (Extreme Value Theory)
- GARCH-EVT hybrid
- Target: 1-day VaR at 99% confidence level

### 4. Validation Requirements
- Validate assumptions for each VaR method
- Risk factor modeling decisions:
  - Direct position value approach vs.
  - Underlying risk factor identification with pricing functions
- Model parameter update frequency analysis
- Impact assessment of update frequency on VaR estimates
- Backtesting implementation for model validation

### 5. Analysis Focus
- Comparative analysis of VaR methodologies
- Visible differentiation in VaR time series across methods
- Justification of modeling choices
- Parameter selection rationale

## Project Goals
1. Implement robust VaR calculation system
2. Compare and contrast different VaR methodologies
3. Validate and backtest risk models
4. Provide comprehensive analysis of results 