# Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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
file_path = os.path.join(project_root, "Raw_Data", "combined_data_10YBond.csv")
print(f"Trying to load data from: {file_path}")
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Print the columns to check
print("Columns in DataFrame:", df.columns)

# Rename columns to be more consistent
df = df.rename(columns={
    "FEDL01 Index  (R1)": "Fed_Funds",
    "USGG10YR Index  (L3)": "10Y_Yield",
    "USGG2YR Index  (L4)": "2Y_Yield",
    "MOVE Index  (R1)": "MOVE"
})

# Convert the date column to datetime format
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", dayfirst=True)
df.set_index("Date", inplace=True)

# Filter data based on date range
df = df.loc[START_DATE:END_DATE]
print(f"Data filtered from {START_DATE} to {END_DATE}")
print(f"Final data shape: {df.shape}")

# Create term spread and drop NAs
df["Term_Spread"] = df["2Y_Yield"] - df["10Y_Yield"]
df = df.dropna(subset=["Fed_Funds", "10Y_Yield", "2Y_Yield", "MOVE", "Term_Spread"])

# Define Target & Risk Factors
target_column = "10Y_Yield"
factor_columns = ["Fed_Funds", "Term_Spread", "MOVE"]

# Verify columns exist
for col in [target_column] + factor_columns:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame. Available columns: {df.columns}")

# Prepare data for modeling
X = df[factor_columns]
y = df[target_column]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Ridge Regression Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = Ridge(alpha=10.0)
model.fit(X_train, y_train)

# Evaluate Performance
y_pred = model.predict(X_scaled)  # predictions for all data
r2_value = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("Ridge alpha:", model.alpha)
print("Combined R²:", r2_value)
print("Combined RMSE:", rmse)

# Store predicted values in dataframe
df["Predicted_Yield"] = y_pred

# Convert yield to price (simplified bond pricing formula)
DURATION = 9  # approximate duration in years
df["Actual_Price"] = 100 / (1 + df[target_column] / 100)**DURATION
df["Synthetic_Price"] = 100 / (1 + df["Predicted_Yield"] / 100)**DURATION

# Compute returns
df["Actual_Returns"] = df["Actual_Price"].pct_change()
df["Synthetic_Returns"] = df["Synthetic_Price"].pct_change()

# Create DataFrame with actual and synthetic values
output_df = pd.DataFrame({
    'Actual_Price': df["Actual_Price"],
    'Synthetic_Price': df["Synthetic_Price"],
    'Actual_Returns': df["Actual_Returns"],
    'Synthetic_Returns': df["Synthetic_Returns"]
}, index=df.index)

# Drop any NaN values (from returns calculation)
output_df = output_df.dropna()

# Save the results to CSV
# Ensure the NAV_returns_Data directory exists
nav_returns_dir = os.path.join(project_root, "NAV_returns_Data")
os.makedirs(nav_returns_dir, exist_ok=True)
output_path = os.path.join(nav_returns_dir, "synthetic_outputs_10ybond.csv")
output_df.to_csv(output_path)
print(f"Synthetic price and returns saved to {output_path}")

# Plot Final Price Prediction x Actual
plt.figure(figsize=(12, 8))
plt.plot(df["Actual_Price"], label="Actual Price", linewidth=2)
plt.plot(df["Synthetic_Price"], label="Synthetic Price", linestyle="--", linewidth=2)
plt.title(f"10Y Bond Price Prediction | Ridge Regression (R² = {r2_value:.4f})", fontsize=12)
plt.xlabel("Date", fontsize=10)
plt.ylabel("Price", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(current_dir, "price_comparison_10ybond.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")
plt.show()

# Print some statistics
print("\nSynthetic Price Statistics:")
print("Correlation with Actual Price:", 
      np.corrcoef(df["Actual_Price"].loc[output_df.index], df["Synthetic_Price"].loc[output_df.index])[0,1])
print("Mean Absolute Error:", 
      np.mean(np.abs(df["Actual_Price"].loc[output_df.index] - df["Synthetic_Price"].loc[output_df.index])))
print("Root Mean Square Error:", 
      np.sqrt(np.mean((df["Actual_Price"].loc[output_df.index] - df["Synthetic_Price"].loc[output_df.index])**2)))
