# Natural Gas Price Forecasting & Storage Contract Valuation

## Overview
This project forecasts natural gas prices using multiple time-series models and uses the forecasted price curve to value a natural gas storage contract.

## Task 1: Price Forecasting
- Models used:
  - Linear Regression (trend)
  - Linear + Sinusoidal model (seasonality)
  - SARIMA
  - Prophet(Model Selected for Valuation)
- Output:
  - 12-month forward price curve

## Task 2: Storage Contract Valuation
- Uses forecasted prices from Task 1
- Models:
  - Injection and withdrawal dates
  - Storage costs
  - Capacity constraints
- Output:
  - Total contract value in USD

## Tools & Libraries
- Python
- pandas, numpy
- statsmodels
- prophet
- matplotlib

## Key Learning
- Time-series forecasting
- Translating forecasts into financial valuation
- Commodity price dynamics
# natural-gas-price-forecasting
Natural gas price forecasting using multiple time-series models and storage contract valuation.
