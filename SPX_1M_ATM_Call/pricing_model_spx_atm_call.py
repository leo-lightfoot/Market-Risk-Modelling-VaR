# Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
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
file_path = os.path.join(project_root, "Raw_Data", "combined_data_spx_atm_call.csv")
print(f"Trying to load data from: {file_path}")
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Print the columns to check
print("Columns in DataFrame:", df.columns)

# Rename columns to be more consistent
df = df.rename(columns={
    "SPX Index  (R1)": "SPX",
    "VIX Index  (R1)": "VIX",
    "MOVE Index  (R1)": "MOVE",
    "USGG10YR Index  (L3)": "Yield_10Y",
    "ATM Call Price": "Option_Price"
})

# Convert the date column to datetime format
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df.set_index("Date", inplace=True)

# Filter data based on date range
df = df.loc[START_DATE:END_DATE]
print(f"Data filtered from {START_DATE} to {END_DATE}")
print(f"Final data shape: {df.shape}")

# Create continuous date range for the entire period
date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
df = df.reindex(date_range)

# Forward fill other columns
df[["SPX", "VIX", "MOVE", "Yield_10Y"]] = df[["SPX", "VIX", "MOVE", "Yield_10Y"]].ffill()

# For Option_Price, first forward fill to get some values, then backward fill for the rest
print(f"Missing values in Option_Price before filling: {df['Option_Price'].isna().sum()}")
df["Option_Price"] = df["Option_Price"].ffill().bfill()
print(f"Missing values in Option_Price after filling: {df['Option_Price'].isna().sum()}")

# Clean Option_Price (convert invalids to NaN)
df["Option_Price"] = pd.to_numeric(df["Option_Price"], errors="coerce")

# Drop any remaining rows with NAs
df = df.dropna(subset=["SPX", "VIX", "MOVE", "Yield_10Y", "Option_Price"])
print(f"Final data shape after cleaning: {df.shape}")

# Define Target & Risk Factors
target_column = "Option_Price"
factor_columns = ["SPX", "VIX", "MOVE", "Yield_10Y"]

# Verify columns exist
for col in [target_column] + factor_columns:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame. Available columns: {df.columns}")

# Prepare data for modeling
X = df[factor_columns].copy()
y = df[target_column]

# Add volatility as feature (rolling std of SPX)
X["SPX_Vol_30d"] = df["SPX"].rolling(30).std().fillna(method='bfill')
X["VIX_Change"] = df["VIX"].pct_change().fillna(0)

# Add interaction terms
X["VIX_MOVE"] = X["VIX"] * X["MOVE"]
X["SPX_VIX"] = X["SPX"] * X["VIX"]

# Train GBM model with more trees and smaller learning rate for better stability
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, 
                                 min_samples_leaf=10, subsample=0.8, random_state=42)
model.fit(X_train, y_train)

# Evaluate Performance
y_pred = model.predict(X)  # predictions for all data
r2_value = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("Combined R²:", r2_value)
print("Combined RMSE:", rmse)

# Store predicted values in dataframe
df["Predicted_Option_Price"] = y_pred

# Apply smoothing to predicted option prices to reduce extreme movements
df["Smoothed_Predicted_Price"] = df["Predicted_Option_Price"].rolling(window=5, center=True).mean()
df["Smoothed_Predicted_Price"] = df["Smoothed_Predicted_Price"].fillna(df["Predicted_Option_Price"])

# Compute returns with caps on extreme values to prevent unrealistic returns
df["Actual_Returns"] = df["Option_Price"].pct_change()
df["Synthetic_Returns"] = df["Smoothed_Predicted_Price"].pct_change()

# Cap extreme returns (limit to +/-30% daily return)
max_return = 0.30
df["Synthetic_Returns"] = df["Synthetic_Returns"].clip(lower=-max_return, upper=max_return)

# Create DataFrame with actual and synthetic values
output_df = pd.DataFrame({
    'Actual_Price': df["Option_Price"],
    'Synthetic_Price': df["Smoothed_Predicted_Price"],
    'Actual_Returns': df["Actual_Returns"],
    'Synthetic_Returns': df["Synthetic_Returns"]
}, index=df.index)

# Drop any NaN values (from returns calculation)
output_df = output_df.dropna()

# Save the results to CSV
# Ensure the NAV_returns_Data directory exists
nav_returns_dir = os.path.join(project_root, "NAV_returns_Data")
os.makedirs(nav_returns_dir, exist_ok=True)
output_path = os.path.join(nav_returns_dir, "synthetic_outputs_spx_atm_call.csv")
output_df.to_csv(output_path)
print(f"Synthetic price and returns saved to {output_path}")

# Plot Final Price Prediction x Actual
plt.figure(figsize=(12, 8))
plt.plot(df["Option_Price"], label="Actual Option Price", linewidth=2)
plt.plot(df["Smoothed_Predicted_Price"], label="Synthetic Option Price", linestyle="--", linewidth=2)
plt.title(f"SPX 1M ATM Call Option Price Prediction | GBR (R² = {r2_value:.4f})", fontsize=12)
plt.xlabel("Date", fontsize=10)
plt.ylabel("Option Price", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(current_dir, "price_comparison_spx_atm_call.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")
plt.show()

# Print some statistics
print("\nSynthetic Price Statistics:")
print("Correlation with Actual Price:", 
      np.corrcoef(df["Option_Price"].loc[output_df.index], df["Smoothed_Predicted_Price"].loc[output_df.index])[0,1])
print("Mean Absolute Error:", 
      np.mean(np.abs(df["Option_Price"].loc[output_df.index] - df["Smoothed_Predicted_Price"].loc[output_df.index])))
print("Root Mean Square Error:", 
      np.sqrt(np.mean((df["Option_Price"].loc[output_df.index] - df["Smoothed_Predicted_Price"].loc[output_df.index])**2)))