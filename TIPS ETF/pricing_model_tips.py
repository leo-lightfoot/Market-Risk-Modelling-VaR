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
file_path = os.path.join(project_root, "Raw_Data", "combined_data_tips.csv")
print(f"Trying to load data from: {file_path}")
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Print the columns to check
print("Columns in DataFrame:", df.columns)

# Convert the date column to datetime format
df["Dates"] = pd.to_datetime(df["Dates"], dayfirst=True)  # Use dayfirst=True for DD-MM-YYYY format
df.set_index("Dates", inplace=True)
df.index.name = 'Date'  # Rename the index to keep consistent format

# Filter data based on date range
df = df.loc[START_DATE:END_DATE]
print(f"Data filtered from {START_DATE} to {END_DATE}")
print(f"Final data shape: {df.shape}")

# Handle missing values in new columns
for col in ['VIX', 'FED_RATE', 'INFL_EXP']:
    if col in df.columns and df[col].isnull().any():
        mean_val = df[col].mean(skipna=True)
        df[col].fillna(mean_val, inplace=True)
        print(f"Filled {df[col].isnull().sum()} missing values in {col} with mean: {mean_val:.4f}")

# Build Features (preserving original logic and adding new features)
features = pd.DataFrame(index=df.index)

# Original features
features["USGG10YR_ret"] = np.log(df["USGG10YR"] / df["USGG10YR"].shift(1))
features["USGGT10Y_ret"] = df["USGGT10Y"].pct_change().clip(-1, 1)
features["Yield_Spread"] = df["USGG10YR"] - df["USGGT10Y"]
features["USGGT10Y_ret_lag1"] = features["USGGT10Y_ret"].shift(1)
features["Yield_Spread_lag1"] = features["Yield_Spread"].shift(1)
features["Ret_Spread_Interaction"] = features["USGGT10Y_ret"] * features["Yield_Spread"]

# New features from additional data points
if 'VIX' in df.columns:
    features["VIX"] = df["VIX"]
    features["VIX_change"] = df["VIX"].pct_change().fillna(0)
    features["VIX_lag1"] = df["VIX"].shift(1)
    
if 'FED_RATE' in df.columns:
    features["FED_RATE"] = df["FED_RATE"]
    features["FED_RATE_change"] = df["FED_RATE"].diff().fillna(0)
    features["FED_RATE_lag1"] = df["FED_RATE"].shift(1)
    
if 'INFL_EXP' in df.columns:
    features["INFL_EXP"] = df["INFL_EXP"]
    features["INFL_EXP_change"] = df["INFL_EXP"].pct_change().fillna(0)
    features["INFL_EXP_lag1"] = df["INFL_EXP"].shift(1)

# Additional interaction terms with new features
if 'INFL_EXP' in df.columns and 'FED_RATE' in df.columns:
    features["INFL_FED_Spread"] = df["INFL_EXP"] - df["FED_RATE"]
    features["Real_Rate_Proxy"] = df["USGG10YR"] - df["INFL_EXP"]

# Define Target column
target_column = "TIP US"

# Compute Log Returns
log_ret_target = np.log(df[target_column] / df[target_column].shift(1))

# Join Target and Features
log_returns = pd.concat([log_ret_target, features], axis=1)
log_returns.columns = [target_column] + list(features.columns)
log_returns.dropna(inplace=True)

# Split Data
X = log_returns.drop(columns=[target_column]).values
y = log_returns[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Ridge Regression
model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
model.fit(X_train, y_train)

# Evaluate Performance
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Print model performance metrics
print("Best alpha:", model.alpha_)
print("Train R²:", r2_score(y_train, y_train_pred))
print("Test R²:", r2_score(y_test, y_test_pred))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Combined metrics for all data
y_pred = model.predict(X)
r2_value = r2_score(y, y_pred)
print("Combined R²:", r2_value)
print("Combined RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

# Print feature importances
feature_names = list(log_returns.drop(columns=[target_column]).columns)
coefficients = list(zip(feature_names, model.coef_))
coefficients.sort(key=lambda x: abs(x[1]), reverse=True)
print("\nTop 10 feature coefficients:")
for feature, coef in coefficients[:10]:
    print(f"{feature}: {coef:.6f}")

# Reconstruct Prices from Log Returns
predicted_returns = model.predict(X)
initial_price = df[target_column].loc[log_returns.index[0]]
predicted_price = initial_price * np.exp(np.cumsum(predicted_returns))
predicted_series = pd.Series(predicted_price, index=log_returns.index)

# Create DataFrame with actual and synthetic values
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
output_path = os.path.join(nav_returns_dir, "synthetic_outputs_tips.csv")
output_df.to_csv(output_path)
print(f"Synthetic NAV and returns saved to {output_path}")

# Plot Final NAV Prediction x Actual
plt.figure(figsize=(12, 8))
plt.plot(df[target_column].loc[log_returns.index], label="Actual NAV", linewidth=2)
plt.plot(predicted_series, label="Synthetic NAV", linestyle="--", linewidth=2)
plt.title(f"TIPS ETF NAV Prediction | Ridge Regression with Enhanced Features (R² = {r2_value:.4f})", fontsize=12)
plt.xlabel("Date", fontsize=10)
plt.ylabel("NAV", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(current_dir, "nav_comparison_tips.png")
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

