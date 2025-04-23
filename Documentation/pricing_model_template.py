############################################################
# ASSET PRICING MODEL TEMPLATE
# 
# This template provides a standardized structure for:
# 1. Loading market data
# 2. Calculating log returns
# 3. Building regression models to create synthetic asset prices
# 4. Evaluating model performance
# 5. Saving results and visualizations
############################################################

# --- SECTION 1: IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

# --- SECTION 2: DATA LOADING & PREPARATION ---
def load_and_prepare_data(file_path, date_column="Date"):
    """
    Load data from CSV and prepare it for modeling
    
    Parameters:
    - file_path: path to the CSV file containing market data
    - date_column: name of the column containing dates
    
    Returns:
    - df: DataFrame with dates as index
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Clean column names (remove whitespace)
    df.columns = df.columns.str.strip()
    
    # Convert dates and set as index
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    
    return df

# --- SECTION 3: RETURN CALCULATION ---
def calculate_returns(df, target_column, factor_columns):
    """
    Calculate log returns for target and factor columns
    
    Parameters:
    - df: DataFrame with price data
    - target_column: column name of the target asset
    - factor_columns: list of column names for risk factors
    
    Returns:
    - log_returns: DataFrame with log returns for all columns
    """
    # Verify columns exist in the DataFrame
    for col in [target_column] + factor_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame. Available columns: {df.columns}")
    
    # Calculate log returns for target
    log_ret_target = np.log(df[target_column] / df[target_column].shift(1))
    
    # Calculate log returns for factors
    log_ret_factors = np.log(df[factor_columns] / df[factor_columns].shift(1))
    
    # Combine into single DataFrame
    log_returns = log_ret_target.to_frame(name=target_column).join(log_ret_factors, how="inner")
    
    # Remove any rows with missing values
    log_returns.dropna(inplace=True)
    
    return log_returns

# --- SECTION 4: FEATURE ENGINEERING ---
def create_interaction_terms(log_returns, factor_columns):
    """
    Create interaction terms between risk factors
    
    Parameters:
    - log_returns: DataFrame with log returns
    - factor_columns: list of column names for risk factors
    
    Returns:
    - log_returns: DataFrame with added interaction terms
    """
    # Create all pairwise interactions
    for i in range(len(factor_columns)):
        for j in range(i+1, len(factor_columns)):
            col1 = factor_columns[i]
            col2 = factor_columns[j]
            # Extract short names for the interaction column
            col1_short = col1.split()[0]
            col2_short = col2.split()[0]
            interaction_name = f"{col1_short}*{col2_short}"
            
            # Create interaction term
            log_returns[interaction_name] = log_returns[col1] * log_returns[col2]
    
    return log_returns

# --- SECTION 5: MODEL TRAINING ---
def train_model(log_returns, target_column, alphas=[0.01, 0.1, 1.0, 10.0, 100.0]):
    """
    Train a Ridge regression model
    
    Parameters:
    - log_returns: DataFrame with log returns and features
    - target_column: name of the target column
    - alphas: list of regularization parameters to try
    
    Returns:
    - model: trained model
    - X: feature matrix
    - y: target vector
    """
    # Prepare features and target
    X = log_returns.drop(columns=[target_column]).values
    y = log_returns[target_column].values
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train appropriate ith cross-validation . in this example it is ridge but make an appropriate choice
    model = RidgeCV(alphas=alphas)
    model.fit(X_train, y_train)
    
    return model, X, y

# --- SECTION 6: MODEL EVALUATION ---
def evaluate_model(model, X, y):
    """
    Evaluate model performance
    
    Parameters:
    - model: trained regression model
    - X: feature matrix
    - y: actual target values
    
    Returns:
    - metrics: dictionary with performance metrics
    - predictions: model predictions
    """
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate metrics
    metrics = {
        "r2": r2_score(y, predictions),
        "rmse": np.sqrt(mean_squared_error(y, predictions))
    }
    
    return metrics, predictions

# --- SECTION 7: NAV RECONSTRUCTION ---
def reconstruct_nav(df, log_returns, target_column, predicted_returns):
    """
    Reconstruct NAV from predicted returns
    
    Parameters:
    - df: original DataFrame with prices
    - log_returns: DataFrame with log returns
    - target_column: name of the target column
    - predicted_returns: array of predicted returns
    
    Returns:
    - predicted_series: Series with predicted NAV values
    """
    # Get initial price (first day in the returns data)
    initial_price = df[target_column].loc[log_returns.index[0]]
    
    # Reconstruct price series using cumulative sum of returns
    predicted_price = initial_price * np.exp(np.cumsum(predicted_returns))
    predicted_series = pd.Series(predicted_price, index=log_returns.index)
    
    return predicted_series

# --- SECTION 8: OUTPUT GENERATION ---
def create_output_df(log_returns, df, target_column, predicted_series, predicted_returns):
    """
    Create output DataFrame with actual and synthetic values
    
    Parameters:
    - log_returns: DataFrame with log returns
    - df: original DataFrame with prices
    - target_column: name of the target column
    - predicted_series: Series with predicted NAV values
    - predicted_returns: array of predicted returns
    
    Returns:
    - output_df: DataFrame with actual and synthetic values
    """
    output_df = pd.DataFrame({
        'Date': log_returns.index,
        'Actual_NAV': df[target_column].loc[log_returns.index],
        'Synthetic_NAV': predicted_series,
        'Actual_Returns': log_returns[target_column],
        'Synthetic_Returns': predicted_returns
    })
    
    return output_df

# --- SECTION 9: VISUALIZATION ---
def create_nav_plot(df, log_returns, target_column, predicted_series, asset_name, output_folder):
    """
    Create and save a plot comparing actual and synthetic NAVs
    
    Parameters:
    - df: original DataFrame with prices
    - log_returns: DataFrame with log returns
    - target_column: name of the target column
    - predicted_series: Series with predicted NAV values
    - asset_name: name of the asset for plot title
    - output_folder: folder to save the plot
    """
    plt.figure(figsize=(12, 8))
    plt.plot(df[target_column].loc[log_returns.index], label="Actual NAV", linewidth=2)
    plt.plot(predicted_series, label="Synthetic NAV", linestyle="--", linewidth=2)
    plt.title(f"{asset_name} NAV Prediction | Log Returns + Interactions (RidgeCV)", fontsize=12)
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("NAV", fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_folder, f"nav_comparison_{asset_name.lower()}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    
    return plot_path

# --- SECTION 10: PERFORMANCE STATISTICS ---
def calculate_statistics(df, log_returns, target_column, predicted_series):
    """
    Calculate performance statistics
    
    Parameters:
    - df: original DataFrame with prices
    - log_returns: DataFrame with log returns
    - target_column: name of the target column
    - predicted_series: Series with predicted NAV values
    
    Returns:
    - stats: dictionary with performance statistics
    """
    actual_values = df[target_column].loc[log_returns.index]
    
    stats = {
        "correlation": np.corrcoef(actual_values, predicted_series)[0,1],
        "mae": np.mean(np.abs(actual_values - predicted_series)),
        "rmse": np.sqrt(np.mean((actual_values - predicted_series)**2))
    }
    
    return stats

# --- SECTION 11: MAIN FUNCTION ---
def run_pricing_model(asset_name, raw_data_path, target_column, factor_columns):
    """
    Run the entire pricing model pipeline
    
    Parameters:
    - asset_name: name of the asset (used for file naming)
    - raw_data_path: path to the raw data CSV file
    - target_column: name of the target column
    - factor_columns: list of column names for risk factors
    """
    # 1. Load and prepare data
    df = load_and_prepare_data(raw_data_path)
    print(f"Loaded data with columns: {df.columns}")
    
    # 2. Calculate returns
    log_returns = calculate_returns(df, target_column, factor_columns)
    print(f"Calculated log returns for {len(log_returns)} days")
    
    # 3. Create interaction terms
    log_returns = create_interaction_terms(log_returns, factor_columns)
    print(f"Created interaction terms, total features: {len(log_returns.columns)-1}")
    
    # 4. Train model
    model, X, y = train_model(log_returns, target_column)
    print(f"Trained model with alpha = {model.alpha_}")
    
    # 5. Evaluate model
    metrics, predicted_returns = evaluate_model(model, X, y)
    print(f"Model performance: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # 6. Reconstruct NAV
    predicted_series = reconstruct_nav(df, log_returns, target_column, predicted_returns)
    
    # 7. Create output DataFrame
    output_df = create_output_df(log_returns, df, target_column, predicted_series, predicted_returns)
    
    # 8. Save output to CSV
    output_path = os.path.join('NAV_returns_Data', f"synthetic_outputs_{asset_name.lower()}.csv")
    output_df.to_csv(output_path)
    print(f"Synthetic NAV and returns saved to {output_path}")
    
    # 9. Create and save plot
    plot_path = create_nav_plot(df, log_returns, target_column, predicted_series, asset_name, asset_name)
    
    # 10. Calculate and print statistics
    stats = calculate_statistics(df, log_returns, target_column, predicted_series)
    print("\nSynthetic NAV Statistics:")
    print(f"Correlation with Actual NAV: {stats['correlation']:.4f}")
    print(f"Mean Absolute Error: {stats['mae']:.4f}")
    print(f"Root Mean Square Error: {stats['rmse']:.4f}")
    
    return output_df, stats

# Example usage:
if __name__ == "__main__":
    # Define asset parameters
    asset_name = "GOLD"  # Used for output file names
    raw_data_path = os.path.join("Raw_Data", "combined_data_gold.csv")
    target_column = "GLD US Equity (USD)"
    factor_columns = ["XAU Curncy", "DXY Curncy", "USGG10Y Index"]
    
    # Run the model
    run_pricing_model(asset_name, raw_data_path, target_column, factor_columns) 