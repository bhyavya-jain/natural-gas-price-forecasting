import pandas as pd
import numpy as np

# Load predicted price curve from Task 1
price_df = pd.read_csv("Price_Curve_Task1.csv")
price_df["Dates"] = pd.to_datetime(price_df["Dates"])
price_df = price_df.sort_values("Dates")

# Price prediction function 
def predict_gas_price(date):
    """
    Returns the predicted gas price for a given date
    using the nearest earlier available price.
    """
    date = pd.to_datetime(date)
    valid_prices = price_df[price_df["Dates"] <= date]

    if valid_prices.empty:
        raise ValueError("No price available before this date")

    return float(valid_prices.iloc[-1]["Prices"])

# Storage contract valuation model 
def calculate_contract_value_model(injection_dates,withdrawal_dates,injection_rate,injection_withdrawal_costs,max_storage_volume,storage_cost_per_month):
    
    total_profit = 0
    avg_num_months = 30.42  # average days in a month

    for i in range(len(injection_dates)):
        injection_price = predict_gas_price(injection_dates[i])
        withdrawal_price = predict_gas_price(withdrawal_dates[i])

        # Convert storage duration into months
        months_in_store = round((pd.to_datetime(withdrawal_dates[i]) - pd.to_datetime(injection_dates[i])).days / avg_num_months)

        # Volume injected (rate + capacity constraint)
        total_injected_volume = min(months_in_store * injection_rate,max_storage_volume)

        # Cost of injection (buy gas + transaction costs)
        cost_of_injection = (total_injected_volume * injection_price -(total_injected_volume / 1_000_000 * injection_withdrawal_costs))

        # Revenue from withdrawal
        revenue_from_sale = total_injected_volume * withdrawal_price

        # Storage cost
        total_storage_cost = months_in_store * storage_cost_per_month

        # Net profit
        total_profit += (revenue_from_sale -cost_of_injection -total_storage_cost)

    return round(total_profit, 2)


# SAMPLE INPUTS 
injection_dates = ["10/31/2024"]
withdrawal_dates = ["2/28/2025"]

injection_rate = 1_000_000          # MMBtu per month
injection_withdrawal_costs = 10_000 # $ per million MMBtu
max_storage_volume = 50_000_000     # MMBtu
storage_cost_per_month = 100_000    # $

# Run valuation
contract_value = calculate_contract_value_model(injection_dates,withdrawal_dates,injection_rate,injection_withdrawal_costs,max_storage_volume,storage_cost_per_month)
print("Storage Contract Value:", contract_value)
