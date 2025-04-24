import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os

# Set date range for analysis
START_DATE = '2004-01-02'
END_DATE = '2025-04-16'

# Load & Prepare Data
# Use absolute path with parent directory to find the Raw_Data folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
file_path = os.path.join(project_root, "Raw_Data", "combined_data_swap.csv")
print(f"Trying to load data from: {file_path}")
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Print the columns to check
print("Columns in DataFrame:", df.columns)

# Rename columns to be more consistent
df = df.rename(columns={
    "USSWIT5 BGN Curncy  (R2)": "Inflation_Swap",
    "USGGBE05 Index  (R3)": "Breakeven_5Y",
    "CPURNSA Index  (R1)": "CPI_YoY",
    "USGG5YR Index  (R4)": "Nominal_5Y_Yield",
    "FEDL01 Index  (R1)": "Fed_Funds"
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

# Backward fill for CPI (monthly data)
df["CPI_YoY"] = df["CPI_YoY"].fillna(method="bfill")

# Forward fill other columns
df[["Inflation_Swap", "Breakeven_5Y", "Nominal_5Y_Yield", "Fed_Funds"]] = df[["Inflation_Swap", "Breakeven_5Y", "Nominal_5Y_Yield", "Fed_Funds"]].ffill()

# Drop NAs
df = df.dropna(subset=["Inflation_Swap", "Breakeven_5Y", "CPI_YoY", "Nominal_5Y_Yield", "Fed_Funds"])
print(f"Final data shape after cleaning: {df.shape}")

# Define Target & Risk Factors
target_column = "Inflation_Swap"
factor_columns = ["Breakeven_5Y", "CPI_YoY", "Nominal_5Y_Yield", "Fed_Funds"]

# Verify columns exist
for col in [target_column] + factor_columns:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame. Available columns: {df.columns}")

# Prepare data for modeling
X = df[factor_columns].copy()
y = df[target_column]

# Regression model fitting
X_scaled = StandardScaler().fit_transform(X)
model = LinearRegression()
model.fit(X_scaled, y)

# Evaluate Performance
y_pred = model.predict(X_scaled)
r2_value = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("Combined R²:", r2_value)
print("Combined RMSE:", rmse)

# Store predicted values in dataframe
df["Predicted_Swap"] = y_pred

# Apply smoothing to predicted swap rates
df["Smoothed_Predicted_Price"] = df["Predicted_Swap"].rolling(window=5, center=True).mean()
df["Smoothed_Predicted_Price"] = df["Smoothed_Predicted_Price"].fillna(df["Predicted_Swap"])

# Calculate swap rate changes and returns using log returns instead of simple returns
df["Actual_Returns"] = np.log(df["Inflation_Swap"] / df["Inflation_Swap"].shift(1))
df["Synthetic_Returns"] = np.log(df["Smoothed_Predicted_Price"] / df["Smoothed_Predicted_Price"].shift(1))

# Cap extreme log returns (approximately equivalent to +/-30% simple returns)
max_log_return = np.log(1.30)  # ~0.262
min_log_return = np.log(0.70)  # ~-0.357
df["Synthetic_Returns"] = df["Synthetic_Returns"].clip(lower=min_log_return, upper=max_log_return)

# Create DataFrame with actual and synthetic values using consistent NAV naming convention
output_df = pd.DataFrame({
    'Actual_NAV': df["Inflation_Swap"],
    'Synthetic_NAV': df["Smoothed_Predicted_Price"],
    'Actual_Returns': df["Actual_Returns"],
    'Synthetic_Returns': df["Synthetic_Returns"]
}, index=df.index)

# Drop any NaN values
output_df = output_df.dropna()

# Save the results to CSV with date column included
# Reset index to make Date a column in the output CSV
output_df_with_date = output_df.reset_index()

# Ensure the NAV_returns_Data directory exists
nav_returns_dir = os.path.join(project_root, "NAV_returns_Data")
os.makedirs(nav_returns_dir, exist_ok=True)
output_path = os.path.join(nav_returns_dir, "synthetic_outputs_swap.csv")
output_df_with_date.to_csv(output_path, index=False)
print(f"Synthetic NAV and returns saved to {output_path}")

# Plot – Actual vs Predicted Swap Rate (only visual output)
plt.figure(figsize=(12, 8))
plt.plot(df.index, df["Inflation_Swap"], label="Actual 5Y ZC Inflation Swap", linewidth=2)
plt.plot(df.index, df["Smoothed_Predicted_Price"], label="Synthetic Inflation Swap (OLS)", linestyle="--", linewidth=2)
plt.title(f"Actual vs Predicted 5Y ZC Inflation Swap Rate | LinearRegression (R² = {r2_value:.4f})", fontsize=12)
plt.xlabel("Date", fontsize=10)
plt.ylabel("Swap Rate (%)", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(current_dir, "price_comparison_inflation_swap.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")
plt.show()

# Print some statistics
print("\nSynthetic NAV Statistics:")
print("Correlation with Actual NAV:", 
      np.corrcoef(df["Inflation_Swap"].loc[output_df.index], df["Smoothed_Predicted_Price"].loc[output_df.index])[0,1])
print("Mean Absolute Error:", 
      np.mean(np.abs(df["Inflation_Swap"].loc[output_df.index] - df["Smoothed_Predicted_Price"].loc[output_df.index])))
print("Root Mean Square Error:", 
      np.sqrt(np.mean((df["Inflation_Swap"].loc[output_df.index] - df["Smoothed_Predicted_Price"].loc[output_df.index])**2)))
