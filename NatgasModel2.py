import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# Load and prepare data
# -----------------------------
natgas_df = pd.read_csv("Nat_Gas.csv")

# Ensure Prices are numeric
natgas_df["Prices"] = pd.to_numeric(natgas_df["Prices"], errors="coerce")

# Drop missing values if any
natgas_df = natgas_df.dropna()

# Monthly index (0, 1, 2, ...)
dates = np.arange(len(natgas_df))
gas_prices = natgas_df["Prices"].values


# -----------------------------
# Model definition
# -----------------------------
def linear_sin_model(x, a, b, c, d, e, f):
    """
    Linear + sinusoidal model
    """
    return a * x + b + c + d * np.sin(2 * np.pi * (e * x + f))


def fit_gas_prices(dates, prices):
    """
    Fit gas price data to linear + sinusoidal model
    """

    start_date = min(dates)
    normalized_dates = np.array(dates) - start_date

    coeffs, _ = curve_fit(linear_sin_model, normalized_dates, prices)

    a, b, c, d, e, f = coeffs

    def predict_gas_price(date):
        normalized_date = date - start_date
        return linear_sin_model(normalized_date, a, b, c, d, e, f)

    return predict_gas_price


# -----------------------------
# Fit model
# -----------------------------
predict_gas_price = fit_gas_prices(dates, gas_prices)

# Predict a future month (example: month 57)
specific_date = 57
predicted_price = predict_gas_price(specific_date)
print("Predicted gas price for month 57:", predicted_price)


# -----------------------------
# Plot results
# -----------------------------
future_dates = np.arange(0, 60)

plt.figure(figsize=(10, 5))
plt.scatter(dates, gas_prices, label="Actual Prices", color="blue")
plt.plot(
    future_dates,
    predict_gas_price(future_dates),
    color="red",
    label="Fitted + Forecast Curve"
)

plt.xlabel("Months since start")
plt.ylabel("Gas Price ($)")
plt.title("Natural Gas Price Forecast (Linear + Seasonal)")
plt.legend()
plt.grid(True)
plt.show()
