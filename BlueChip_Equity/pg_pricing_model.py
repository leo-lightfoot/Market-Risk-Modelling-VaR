# Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

# Set date range for analysis
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

# Load & Prepare Data
# Use absolute path with parent directory to find the Raw_Data folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
file_path = os.path.join(project_root, "Raw_Data", "combined_data_pg.csv")
print(f"Trying to load data from: {file_path}")
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Print the columns to check
print("Columns in DataFrame:", df.columns)

# Convert the date column to datetime format with the correct format parameter
# Using dayfirst=True since the format appears to be DD-MM-YYYY
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df.set_index("Date", inplace=True)

# Filter data based on date range
df = df.loc[START_DATE:END_DATE]
print(f"Data filtered from {START_DATE} to {END_DATE}")
print(f"Final data shape: {df.shape}")

# Define Target & Risk Factors
target_column = "PG US Equity"
# Remove dividend yield from factor columns as it will be handled separately
factor_columns = ["S&P 500 Index", "VIX Index", "USD Index", "Treasury Rate Proxy", 
                 "XLP Sector Index", "PG P/E Ratio"]

# Handle missing values in any columns
for col in factor_columns + ["DIVIDEND_YIELD"]:
    if df[col].isnull().any():
        mean_val = df[col].mean(skipna=True)
        df[col].fillna(mean_val, inplace=True)
        print(f"Filled {df[col].isnull().sum()} missing values in {col} with mean: {mean_val:.4f}")

# Verify columns exist
for col in [target_column] + factor_columns + ["DIVIDEND_YIELD"]:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame. Available columns: {df.columns}")

# Compute Log Returns for all columns (except dividend yield)
log_ret_target = np.log(df[target_column] / df[target_column].shift(1))
log_ret_factors = np.log(df[factor_columns] / df[factor_columns].shift(1))
log_returns = log_ret_target.to_frame(name=target_column).join(log_ret_factors, how="inner")

# Add dividend yield as a direct factor (not log-transformed)
log_returns["DIVIDEND_YIELD"] = df["DIVIDEND_YIELD"][log_returns.index]

# Create interaction terms (key pairwise combinations)
# Market factor interactions
log_returns["S&P*VIX"] = log_returns["S&P 500 Index"] * log_returns["VIX Index"]
log_returns["S&P*XLP"] = log_returns["S&P 500 Index"] * log_returns["XLP Sector Index"]
log_returns["VIX*USD"] = log_returns["VIX Index"] * log_returns["USD Index"]
log_returns["USD*Treasury"] = log_returns["USD Index"] * log_returns["Treasury Rate Proxy"]

# Company-specific factor interactions
log_returns["P/E*XLP"] = log_returns["PG P/E Ratio"] * log_returns["XLP Sector Index"]
log_returns["P/E*S&P"] = log_returns["PG P/E Ratio"] * log_returns["S&P 500 Index"]
log_returns["P/E*VIX"] = log_returns["PG P/E Ratio"] * log_returns["VIX Index"]
# Use raw dividend yield for interactions
log_returns["DIV*S&P"] = log_returns["DIVIDEND_YIELD"] * log_returns["S&P 500 Index"]
log_returns["DIV*XLP"] = log_returns["DIVIDEND_YIELD"] * log_returns["XLP Sector Index"]

log_returns.dropna(inplace=True)

# Train Ridge Regression Model
X = log_returns.drop(columns=[target_column]).values
y = log_returns[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
model.fit(X_train, y_train)

# Evaluate Performance
y_pred = model.predict(X)  # predictions for all data
r2_value = r2_score(y, y_pred)
print("Best alpha:", model.alpha_)
print("Combined R²:", r2_value)
print("Combined RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

# Reconstruct Prices from Log Returns
predicted_returns = model.predict(X)
initial_price = df[target_column].loc[log_returns.index[0]]
predicted_price = initial_price * np.exp(np.cumsum(predicted_returns))
predicted_series = pd.Series(predicted_price, index=log_returns.index)

# Create DataFrame with actual and synthetic values - Fix to avoid duplicate Date column
output_df = pd.DataFrame({
    'Actual_NAV': df[target_column].loc[log_returns.index],
    'Synthetic_NAV': predicted_series,
    'Actual_Returns': log_returns[target_column],
    'Synthetic_Returns': predicted_returns
}, index=log_returns.index)

# Save the results to CSV
# Ensure the NAV_returns_Data directory exists
nav_returns_dir = os.path.join(project_root, "NAV_returns_Data")
os.makedirs(nav_returns_dir, exist_ok=True)
output_path = os.path.join(nav_returns_dir, "synthetic_outputs_pg.csv")
output_df.to_csv(output_path)
print(f"Synthetic NAV and returns saved to {output_path}")

# Plot Final NAV Prediction x Actual
plt.figure(figsize=(12, 8))
plt.plot(df[target_column].loc[log_returns.index], label="Actual NAV", linewidth=2)
plt.plot(predicted_series, label="Synthetic NAV", linestyle="--", linewidth=2)
plt.title(f"PG NAV Prediction | Log Returns + Interactions (RidgeCV) (R² = {r2_value:.4f})", fontsize=12)
plt.xlabel("Date", fontsize=10)
plt.ylabel("NAV", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(current_dir, "nav_comparison_pg.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")
plt.show()

# Print some statistics
print("\nSynthetic NAV Statistics:")
print("Correlation with Actual NAV:", 
      np.corrcoef(df[target_column].loc[log_returns.index], predicted_series)[0,1])
print("Mean Absolute Error:", 
      np.mean(np.abs(df[target_column].loc[log_returns.index] - predicted_series)))
print("Root Mean Square Error:", 
      np.sqrt(np.mean((df[target_column].loc[log_returns.index] - predicted_series)**2))) 