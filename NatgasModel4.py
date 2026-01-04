# Time Series Model- PROPHET MODEL
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# --------------------------------------------------
# 1. Load and prepare data
# --------------------------------------------------
natgas_df = pd.read_csv("Nat_Gas.csv")

# Convert date column
natgas_df["Dates"] = pd.to_datetime(
    natgas_df["Dates"],
    format="%m/%d/%y"
)

# Prophet requires columns named 'ds' and 'y'
prophet_df = natgas_df.rename(
    columns={"Dates": "ds", "Prices": "y"}
)

# Ensure numeric prices
prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce")
prophet_df = prophet_df.dropna()

# --------------------------------------------------
# 2. Fit Prophet Model
# --------------------------------------------------
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

model.fit(prophet_df)

# --------------------------------------------------
# 3. Create future dates (12 months ahead)
# --------------------------------------------------
future = model.make_future_dataframe(
    periods=12,
    freq="ME"
)

forecast = model.predict(future)

# --------------------------------------------------
# 4. Plot forecast
# --------------------------------------------------
model.plot(forecast)
plt.title("Natural Gas Price Forecast (Prophet)")
plt.xlabel("Year")
plt.ylabel("Gas Price ($)")
plt.show()

# --------------------------------------------------
# 5. Plot trend & seasonality components
# --------------------------------------------------
model.plot_components(forecast)
plt.show()

# --------------------------------------------------
# 6. Print forecasted 2025 prices
# --------------------------------------------------
forecast_2025 = forecast[forecast["ds"].dt.year == 2025][["ds", "yhat"]]
print("Forecasted Gas Prices for 2025:")
print(forecast_2025.round(2))

# 7. Save price curve for Task 2 (REQUIRED)
# --------------------------------------------------
price_curve = forecast[["ds", "yhat"]].rename(
    columns={"ds": "Dates", "yhat": "Prices"}
)

price_curve.to_csv("Price_Curve_Task1.csv", index=False)

print("Task 1 complete: Price_Curve_Task1.csv generated successfully.")