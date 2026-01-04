# Time Series Model SARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --------------------------------------------------
# 1. Load and prepare data
# --------------------------------------------------
natgas_df = pd.read_csv("Nat_Gas.csv")

# Convert Dates to datetime
natgas_df["Dates"] = pd.to_datetime(
    natgas_df["Dates"],
    format="%m/%d/%y"
)

# Ensure numeric prices
natgas_df["Prices"] = pd.to_numeric(natgas_df["Prices"], errors="coerce")
natgas_df = natgas_df.dropna()

# Set Date index
natgas_df = natgas_df.set_index("Dates")
natgas_df = natgas_df.sort_index()

# Ensure monthly frequency
natgas_df = natgas_df.asfreq("M")

# --------------------------------------------------
# 2. Fit SARIMA Model
# --------------------------------------------------
# (p,d,q) x (P,D,Q,seasonal_period)
model = SARIMAX(
    natgas_df["Prices"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)

# --------------------------------------------------
# 3. Forecast future prices (12 months = 2025)
# --------------------------------------------------
forecast_steps = 12
forecast = results.get_forecast(steps=forecast_steps)

forecast_index = pd.date_range(
    start=natgas_df.index[-1] + pd.offsets.MonthEnd(1),
    periods=forecast_steps,
    freq="M"
)

forecast_df = pd.DataFrame({
    "Forecast": forecast.predicted_mean
}, index=forecast_index)

# --------------------------------------------------
# 4. Plot actuals + forecast (continuous)
# --------------------------------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    natgas_df.index,
    natgas_df["Prices"],
    label="Actual Prices",
    color="blue"
)

plt.plot(
    forecast_df.index,
    forecast_df["Forecast"],
    label="Forecast (Time Series)",
    color="orange"
)

plt.xlabel("Year")
plt.ylabel("Gas Price ($)")
plt.title("Natural Gas Price Forecast (SARIMA)")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# 5. Print forecast values
# --------------------------------------------------
print("Forecasted Gas Prices:")
print(forecast_df.round(2))
