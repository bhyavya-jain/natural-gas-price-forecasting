import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# --------------------------------------------------
# 1. LOAD & PREP DATA
# --------------------------------------------------
natgas_df = pd.read_csv('Nat_Gas.csv')

natgas_df['Dates'] = pd.to_datetime(
    natgas_df['Dates'],
    format='%m/%d/%y'
)

natgas_df['Year'] = natgas_df['Dates'].dt.year
natgas_df['Month'] = natgas_df['Dates'].dt.month

# --------------------------------------------------
# 2. FORECAST FUNCTION (NO WARNINGS)
# --------------------------------------------------
def next_year_price(next_year):
    prices = []

    for month in range(1, 13):
        month_data = natgas_df[natgas_df['Month'] == month]

        X = month_data[['Year']]
        y = month_data['Prices']

        model = LinearRegression()
        model.fit(X, y)

        pred = model.predict(pd.DataFrame({'Year': [next_year]}))[0]
        prices.append(round(pred, 2))

    return prices


gas_price25 = next_year_price(2025)

# --------------------------------------------------
# 3. MONTH-END DATES FOR 2025
# --------------------------------------------------
def get_last_of_each_month(year):
    dates = []
    current_date = datetime(year, 12, 31)

    while current_date.year == year:
        dates.append(current_date)
        current_date = current_date.replace(day=1) - timedelta(days=1)

    return dates[::-1]


dates_2025 = get_last_of_each_month(2025)

# --------------------------------------------------
# 4. FORECAST DATAFRAME
# --------------------------------------------------
forecast_df = pd.DataFrame({
    'Dates': dates_2025,
    'Prices': gas_price25
})

# --------------------------------------------------
# 5. HISTORICAL DATAFRAME
# --------------------------------------------------
historical_df = natgas_df[['Dates', 'Prices']].copy()

# --------------------------------------------------
# 6. CREATE CONTINUITY BRIDGE
# --------------------------------------------------
last_actual_date = historical_df['Dates'].iloc[-1]
last_actual_price = historical_df['Prices'].iloc[-1]

forecast_dates_continuous = pd.concat([
    pd.Series([last_actual_date]),
    forecast_df['Dates']
])

forecast_prices_continuous = pd.concat([
    pd.Series([last_actual_price]),
    forecast_df['Prices']
])

# --------------------------------------------------
# 7. FINAL CONTINUOUS PLOT (DIFFERENT COLORS)
# --------------------------------------------------
plt.figure(figsize=(10, 5))

# Actuals
plt.plot(
    historical_df['Dates'],
    historical_df['Prices'],
    label='Actuals (2021–2024)',
    linewidth=2
)

# Forecast (continuous + different color)
plt.plot(
    forecast_dates_continuous,
    forecast_prices_continuous,
    label='Forecast 2025',
    linestyle='--',
    marker='o',
    linewidth=2
)

# Forecast start marker
plt.axvline(
    forecast_df['Dates'].iloc[0],
    linestyle=':',
    alpha=0.6,
    label='Forecast Start'
)

plt.xlabel('Year')
plt.ylabel('Gas Price ($)')
plt.title('Natural Gas Price Forecast', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()
