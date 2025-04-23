import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os

# Set date range for analysis
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'

# Load & Prepare Data
# Use absolute path with parent directory to find the Raw_Data folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
file_path = os.path.join(project_root, "Raw_Data", "combined_data_privatecredit.csv")
print(f"Trying to load data from: {file_path}")
df = pd.read_csv(file_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Print the columns to check
print("Columns in DataFrame:", df.columns)

# Convert the unnamed date column to datetime and set as index
df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"])
df.set_index("Unnamed: 0", inplace=True)
df.index.name = 'Date'  # Rename the index to 'Date'

# Filter date range
df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]
print(f"Data filtered from {START_DATE} to {END_DATE}")
print(f"Final data shape: {df.shape}")

# Define features and target
target_column = 'BIZD'
features = [
    'XLF_Change', 'IWM_Change', 'HYG', 'HY_OAS_Change', 'VIX_Change',
    'XLF_Change_HighVol', 'IWM_Change_HighVol', 'HYG_HighVol',
    'XLF_Change_Stress', 'IWM_Change_Stress', 'HYG_Stress',
    'XLF_WideCred', 'HYG_WideCred',
    'Credit_Market_Stress', 'Rate_Credit_Stress'
]

# Verify columns exist
for col in [target_column] + features:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame. Available columns: {df.columns}")

# Clean data
df_clean = df.dropna(subset=features + [target_column])
print(f"Using {len(df_clean)} rows after removing missing data.")

# Prepare data
X = df_clean[features]
y = df_clean[target_column]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train ElasticNet Model
model = ElasticNetCV(
    l1_ratio=[.1, .5, .7, .9],
    alphas=np.logspace(-4, 4, 50),
    cv=TimeSeriesSplit(n_splits=3),
    max_iter=2000,
    random_state=42
)
model.fit(X_scaled, y)

# Evaluate Performance
predicted_returns = model.predict(X_scaled)
r2_value = model.score(X_scaled, y)
print("Model R²:", r2_value)
print("RMSE:", np.sqrt(np.mean((y - predicted_returns)**2)))

# Print top feature coefficients
coefs = list(zip(features, model.coef_))
coefs.sort(key=lambda x: abs(x[1]), reverse=True)
print("\nTop 5 feature coefficients:")
for feature, coef in coefs[:5]:
    print(f"{feature}: {coef:.4f}")

# Reconstruct NAVs from Returns
initial_price = 100
predicted_price = initial_price * np.exp(np.cumsum(predicted_returns))
predicted_series = pd.Series(predicted_price, index=df_clean.index)
actual_price = initial_price * np.exp(np.cumsum(df_clean[target_column]))
actual_series = pd.Series(actual_price, index=df_clean.index)

# Create DataFrame with actual and synthetic values
output_df = pd.DataFrame({
    'Actual_NAV': actual_series,
    'Synthetic_NAV': predicted_series,
    'Actual_Returns': df_clean[target_column],
    'Synthetic_Returns': predicted_returns
}, index=df_clean.index)

# Save the results to CSV
# Ensure the NAV_returns_Data directory exists
nav_returns_dir = os.path.join(project_root, "NAV_returns_Data")
os.makedirs(nav_returns_dir, exist_ok=True)
output_path = os.path.join(nav_returns_dir, "synthetic_outputs_pc.csv")
output_df.to_csv(output_path)
print(f"Synthetic NAV and returns saved to {output_path}")

# Plot Final NAV Prediction x Actual
plt.figure(figsize=(12, 8))
plt.plot(actual_series, label="Actual NAV", linewidth=2)
plt.plot(predicted_series, label="Synthetic NAV", linestyle="--", linewidth=2)
plt.title(f"Private Credit NAV Prediction | ElasticNet with Feature Interactions (R² = {r2_value:.4f})", fontsize=12)
plt.xlabel("Date", fontsize=10)
plt.ylabel("NAV", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(current_dir, "nav_comparison_pc.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")
plt.show()

# Print some statistics
print("\nSynthetic NAV Statistics:")
print("Correlation with Actual NAV:", 
      np.corrcoef(actual_series, predicted_series)[0,1])
print("Mean Absolute Error:", 
      np.mean(np.abs(actual_series - predicted_series)))
print("Root Mean Square Error:", 
      np.sqrt(np.mean((actual_series - predicted_series)**2))) 