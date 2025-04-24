# Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os
from datetime import timedelta
from scipy.stats import norm

# Black-Scholes Option Pricing Function
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes price for a call option
    
    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free interest rate (annual)
    sigma: Volatility of the underlying asset
    
    Returns:
    Call option price
    """
    # Ensure inputs are valid
    S = max(S, 0.01)  # Ensure price is positive
    K = max(K, 0.01)  # Ensure strike is positive
    T = max(T, 0.001)  # Minimum time to expiration (1/1000 of a year ≈ 0.4 days)
    sigma = max(sigma, 0.001)  # Minimum volatility
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price

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

# Generate actual SPX option expiration dates (3rd Friday of each month)
def generate_monthly_expiration_dates(start_date, end_date):
    """Generate option expiration dates (3rd Friday of each month)"""
    expiry_dates = []
    
    # Convert string dates to datetime objects if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Start from the first day of the month of start_date
    current_date = pd.Timestamp(year=start_date.year, month=start_date.month, day=1)
    
    while current_date <= end_date:
        # Find the third Friday of this month
        # Start with the first day of the month
        day = current_date
        
        # Find the first Friday
        while day.weekday() != 4:  # 4 is Friday (0-based, Monday=0)
            day += timedelta(days=1)
        
        # Move to the third Friday
        third_friday = day + timedelta(days=14)
        
        # Add to list if within range
        if third_friday >= start_date and third_friday <= end_date:
            expiry_dates.append(third_friday)
        
        # Move to next month
        if current_date.month == 12:
            current_date = pd.Timestamp(year=current_date.year + 1, month=1, day=1)
        else:
            current_date = pd.Timestamp(year=current_date.year, month=current_date.month + 1, day=1)
    
    return expiry_dates

# Generate expiration dates
expiration_dates = generate_monthly_expiration_dates(df.index.min(), df.index.max())
print(f"Generated {len(expiration_dates)} monthly expiration dates")

# Calculate time to expiration for each date
def calculate_time_to_expiration(dates, expiry_dates):
    """Calculate time to next expiration for each date in the index"""
    tte = []
    
    for date in dates:
        # Find the next expiration date
        next_expiry = None
        for expiry in expiry_dates:
            if expiry >= date:
                next_expiry = expiry
                break
        
        if next_expiry is None:
            # If no future expiry, use last available
            tte.append(0.01)  # Small non-zero value
        else:
            # Calculate time to expiration in years
            days_to_expiry = (next_expiry - date).days
            years_to_expiry = days_to_expiry / 365.0
            tte.append(max(years_to_expiry, 0.01))  # Ensure minimum value
    
    return tte

# Add time to expiration feature
df['Time_To_Expiry'] = calculate_time_to_expiration(df.index, expiration_dates)

# Add Black-Scholes theoretical price as feature
df['BS_Theoretical'] = np.nan

# Calculate Black-Scholes price for each row
for i in range(len(df)):
    # Parameters for Black-Scholes
    S = df['SPX'].iloc[i]  # Current SPX price
    K = S  # For ATM options, strike = current price
    T = df['Time_To_Expiry'].iloc[i]  # Time to expiration in years
    r = df['Yield_10Y'].iloc[i] / 100.0  # Convert from percentage to decimal
    sigma = df['VIX'].iloc[i] / 100.0  # Convert VIX from percentage to decimal
    
    # Calculate theoretical price
    try:
        bs_price = black_scholes_call(S, K, T, r, sigma)
        df.loc[df.index[i], 'BS_Theoretical'] = bs_price
    except Exception as e:
        print(f"Error calculating BS price for index {i}: {e}")
        # Use previous value or set to NaN
        if i > 0:
            df.loc[df.index[i], 'BS_Theoretical'] = df['BS_Theoretical'].iloc[i-1]
        else:
            df.loc[df.index[i], 'BS_Theoretical'] = np.nan

# Fill any missing BS values
df['BS_Theoretical'] = df['BS_Theoretical'].fillna(method='ffill').fillna(method='bfill')

# Add moneyness feature
df['Moneyness'] = 1.0  # ATM options are defined as having moneyness = 1

# Define Target & Risk Factors
target_column = "Option_Price"
factor_columns = ["SPX", "VIX", "MOVE", "Yield_10Y", "BS_Theoretical", "Time_To_Expiry", "Moneyness"]

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
X["BS_Theo_TTE"] = X["BS_Theoretical"] / (X["Time_To_Expiry"] + 0.01)  # BS price divided by time to expiry

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

# Print feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Compute raw log returns
df["Raw_Actual_Returns"] = np.log(df["Option_Price"] / df["Option_Price"].shift(1))
df["Raw_Synthetic_Returns"] = np.log(df["Smoothed_Predicted_Price"] / df["Smoothed_Predicted_Price"].shift(1))

# Implement Option Roll-Over Logic
# Use real expiration dates instead of simplified 30-day cycle
print("\nImplementing option roll-over logic with actual expiration dates...")

# Generate roll dates (1 week before expiration)
roll_dates = [expiry - timedelta(days=7) for expiry in expiration_dates]

# Initialize arrays for total return index calculation
initial_investment = 100.0  # Start with $100 investment
first_actual_price = df["Option_Price"].iloc[0]
num_contracts_held = initial_investment / df["Smoothed_Predicted_Price"].iloc[0]

# Arrays to store the total return index and adjusted returns
positions = np.zeros(len(df))
total_return_index = np.zeros(len(df))
roll_indicators = np.zeros(len(df), dtype=bool)

# First day initialization
positions[0] = num_contracts_held
total_return_index[0] = initial_investment

# Identify the closest actual dates in our dataframe to the theoretical roll dates
actual_roll_dates = []
for roll_date in roll_dates:
    # Find the closest date that exists in our dataframe
    closest_date = df.index[df.index.get_indexer([roll_date], method='nearest')[0]]
    actual_roll_dates.append(closest_date)

# Print roll dates for verification
print(f"Generated {len(actual_roll_dates)} option roll dates")

# Calculate total return index with roll-over adjustments
for i in range(1, len(df)):
    current_date = df.index[i]
    prev_date = df.index[i-1]
    
    # Check if this is a roll date
    is_roll_date = current_date in actual_roll_dates
    
    if is_roll_date:
        # On roll dates: sell existing position and buy new contracts
        roll_indicators[i] = True
        
        # Calculate current value of position
        current_value = positions[i-1] * df["Smoothed_Predicted_Price"].iloc[i]
        
        # Buy new position with full proceeds (reinvest)
        positions[i] = current_value / df["Smoothed_Predicted_Price"].iloc[i]
        
        # Total return index carries forward
        total_return_index[i] = current_value
        
        print(f"Roll at {current_date}: Position value = ${current_value:.2f}")
    else:
        # Regular day: position size stays the same, value changes with price
        positions[i] = positions[i-1]
        current_value = positions[i] * df["Smoothed_Predicted_Price"].iloc[i]
        total_return_index[i] = current_value

# Store results in DataFrame
df["Roll_Indicator"] = roll_indicators
df["Option_Contracts"] = positions
df["TR_Index"] = total_return_index

# Calculate returns based on the total return index
df["Actual_Returns"] = df["Raw_Actual_Returns"].copy()  # Keep original for actual
df["Synthetic_Returns"] = np.log(df["TR_Index"] / df["TR_Index"].shift(1)).fillna(0)

# Cap extreme log returns (approximately equivalent to +/-30% simple returns)
max_log_return = np.log(1.30)  # ~0.262
min_log_return = np.log(0.70)  # ~-0.357
df["Synthetic_Returns"] = df["Synthetic_Returns"].clip(lower=min_log_return, upper=max_log_return)

# Normalize both series to the same starting point
# Get the first non-NA values for both series
first_actual_nav = df["Option_Price"].dropna().iloc[0]
first_synthetic_tr = df["TR_Index"].dropna().iloc[0]

# Create normalized versions
df["Normalized_Actual_NAV"] = df["Option_Price"] * (100 / first_actual_nav)
df["Normalized_Synthetic_NAV"] = df["TR_Index"] * (100 / first_synthetic_tr)

# Create DataFrame with actual and synthetic values using consistent NAV naming convention
output_df = pd.DataFrame({
    'Actual_NAV': df["Option_Price"],
    'Synthetic_NAV': df["TR_Index"],
    'Normalized_Actual_NAV': df["Normalized_Actual_NAV"],
    'Normalized_Synthetic_NAV': df["Normalized_Synthetic_NAV"],
    'BS_Theoretical': df["BS_Theoretical"],
    'Actual_Returns': df["Actual_Returns"],
    'Synthetic_Returns': df["Synthetic_Returns"],
    'Time_To_Expiry': df["Time_To_Expiry"]
}, index=df.index)

# Drop any NaN values (from returns calculation)
output_df = output_df.dropna()

# Save the results to CSV with date column included
# Reset index to make Date a column in the output CSV
output_df_with_date = output_df.reset_index()

# Ensure the NAV_returns_Data directory exists
nav_returns_dir = os.path.join(project_root, "NAV_returns_Data")
os.makedirs(nav_returns_dir, exist_ok=True)
output_path = os.path.join(nav_returns_dir, "synthetic_outputs_spx_atm_call.csv")
output_df_with_date.to_csv(output_path, index=False)
print(f"Synthetic NAV and returns saved to {output_path}")

# Plot Final Price Prediction x Actual vs Total Return Index
plt.figure(figsize=(12, 8))

# Plot 1: Raw prices
plt.subplot(2, 1, 1)
plt.plot(df.index, df["Option_Price"], label="Actual Option Price", alpha=0.5, linewidth=1)
plt.plot(df.index, df["Smoothed_Predicted_Price"], label="Synthetic Option Price", alpha=0.5, linestyle="--", linewidth=1)
plt.plot(df.index, df["TR_Index"], label="Total Return Index (with Roll-Overs)", color='green', linewidth=2)
plt.plot(df.index, df["BS_Theoretical"], label="Black-Scholes Theoretical", color='purple', linestyle=':', linewidth=1)

# Mark roll dates
for roll_date in actual_roll_dates:
    if roll_date in df.index:
        plt.axvline(x=roll_date, color='gray', linestyle=':', alpha=0.5)

plt.title(f"SPX 1M ATM Call Option - Raw Price vs Total Return Index | GBR (R² = {r2_value:.4f})", fontsize=12)
plt.ylabel("Price/Value", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True)

# Plot 2: Normalized prices
plt.subplot(2, 1, 2)
plt.plot(df.index, df["Normalized_Actual_NAV"], label="Normalized Actual NAV", color='blue', linewidth=2)
plt.plot(df.index, df["Normalized_Synthetic_NAV"], label="Normalized Synthetic NAV", color='red', linewidth=2)

# Mark roll dates
for roll_date in actual_roll_dates:
    if roll_date in df.index:
        plt.axvline(x=roll_date, color='gray', linestyle=':', alpha=0.5)

plt.title("Normalized NAVs (Both starting at 100)", fontsize=12)
plt.xlabel("Date", fontsize=10)
plt.ylabel("Normalized Value", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True)

plt.tight_layout()

# Save the plot
plot_path = os.path.join(current_dir, "price_comparison_spx_atm_call.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_path}")
plt.show()

# Print some statistics
print("\nSynthetic NAV Statistics:")
print("Correlation with Actual NAV:", 
      np.corrcoef(df["Option_Price"].loc[output_df.index], df["TR_Index"].loc[output_df.index])[0,1])
print("Correlation with Normalized NAVs:", 
      np.corrcoef(df["Normalized_Actual_NAV"].loc[output_df.index], df["Normalized_Synthetic_NAV"].loc[output_df.index])[0,1])
print("Initial Option Price:", df["Smoothed_Predicted_Price"].iloc[0])
print("Final Total Return Index:", df["TR_Index"].iloc[-1])
print("Number of Rolls:", len(actual_roll_dates))
print("Total Return:", (df["TR_Index"].iloc[-1] / initial_investment - 1) * 100, "%")
print("Total Return:", (df["TR_Index"].iloc[-1] / initial_investment - 1) * 100, "%")