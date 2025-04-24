import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Incorporate the Portfolio class directly instead of importing
class Portfolio:
    def __init__(self, initial_capital=1000000.0, start_date=None):
        """Initialize portfolio with initial capital and start date"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.start_date = pd.to_datetime(start_date) if start_date else datetime.now()
        self.allocation = {}
        self.holdings = {}
        self.instruments = {}
        self.nav_history = pd.DataFrame(columns=['Date', 'NAV'])
        self.returns_history = pd.DataFrame(columns=['Date', 'Return'])
        self._load_pricing_data()
        
    def _load_instrument_data(self, instrument_name, file_name):
        """Load instrument data from NAV_returns_Data directory
        
        Args:
            instrument_name: Name to use for the instrument in the portfolio
            file_name: CSV file name in NAV_returns_Data directory
            
        Returns:
            True if successful, False otherwise
        """
        data_dir = os.path.join(os.getcwd(), 'NAV_returns_Data')
        try:
            pricing_file = os.path.join(data_dir, file_name)
            self.instruments[instrument_name] = pd.read_csv(pricing_file, index_col=0, parse_dates=True)
            self.instruments[instrument_name] = self.instruments[instrument_name].asfreq('D', method='ffill')
            print(f"Loaded {instrument_name} data with {len(self.instruments[instrument_name])} observations")
            return True
        except FileNotFoundError:
            print(f"Error: {instrument_name} pricing data not found at {pricing_file}")
            return False
        
    def _load_pricing_data(self):
        """Load instrument pricing data from NAV_returns_Data directory"""
        # Dictionary mapping instrument names to their data files
        instrument_files = {
            'Gold': 'synthetic_outputs_gold.csv',
            'Apple': 'synthetic_outputs_apple.csv',
            'JnJ': 'synthetic_outputs_jnj.csv',
            'PG': 'synthetic_outputs_pg.csv',
            'EM': 'synthetic_outputs_em.csv',
            'Private Credit': 'synthetic_outputs_pc.csv',
            'TIPS ETF': 'synthetic_outputs_tips.csv',
            '10Y Bond': 'synthetic_outputs_10ybond.csv',
            'SPX ATM Call': 'synthetic_outputs_spx_atm_call.csv',
            'Inflation Swap': 'synthetic_outputs_swap.csv',
            'Convertible Bond': 'synthetic_outputs_cwb.csv'
        }
        
        # Load all instruments
        for instrument_name, file_name in instrument_files.items():
            self._load_instrument_data(instrument_name, file_name)
        
        # Align datasets to common date range if they exist
        self._align_datasets()
        
    
    def _align_datasets(self):
        """Align all instrument datasets to a common date range"""
        if not self.instruments:
            return
            
        # Find common date range across all instruments
        start_dates = []
        end_dates = []
        
        for name, data in self.instruments.items():
            if data is not None and not data.empty:
                start_dates.append(data.index.min())
                end_dates.append(data.index.max())
        
        if not start_dates or not end_dates:
            return
            
        start_date = max(start_dates)
        end_date = min(end_dates)
        
        # If user provided start date is after the earliest data point, use that instead
        if self.start_date > start_date:
            start_date = self.start_date
            
        # Create a common daily date range
        daily_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Reindex all datasets to the same daily range
        for name, data in self.instruments.items():
            if data is not None and not data.empty:
                self.instruments[name] = data.reindex(daily_range).fillna(method='ffill')
        
        print(f"Datasets aligned with {len(daily_range)} daily observations from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    def add_instrument(self, name, data_path=None, data=None):
        """Add a new instrument to the portfolio
        
        Args:
            name: Name of the instrument
            data_path: Path to CSV data file (alternative to data)
            data: DataFrame with instrument data (alternative to data_path)
        """
        if data is not None:
            self.instruments[name] = data
        elif data_path is not None:
            try:
                self.instruments[name] = pd.read_csv(data_path, index_col=0, parse_dates=True)
                self.instruments[name] = self.instruments[name].asfreq('D', method='ffill')
                print(f"Loaded {name} data with {len(self.instruments[name])} observations")
            except FileNotFoundError:
                print(f"Error: {name} data file not found at {data_path}")
        else:
            print("Error: Either data_path or data must be provided")
            
        # Re-align datasets
        self._align_datasets()
        
        # Add to allocation with zero allocation initially
        if name not in self.allocation:
            self.allocation[name] = 0.0
            self.holdings[name] = 0.0

    def set_allocation(self, **allocations):
        """Set portfolio allocation between instruments
        
        Args:
            **allocations: Keyword arguments with instrument name and percentage
                           e.g., Private_Credit=50.0, Gold=25.0, Cash=25.0
        """
        # Validate percentages sum to 100
        total_pct = sum(allocations.values())
        if abs(total_pct - 100.0) > 0.001:
            raise ValueError(f"Allocation percentages must sum to 100% (got {total_pct}%)")
            
        # Set allocation percentages
        for name, pct in allocations.items():
            self.allocation[name] = pct / 100.0
        
        # Calculate holdings based on allocation
        for name, alloc in self.allocation.items():
            self.holdings[name] = self.current_capital * alloc
        
        print(f"Portfolio allocation set to: {', '.join([f'{k}: {v}%' for k, v in allocations.items()])}")
        return self.allocation
    
    def calculate_returns(self, end_date=None, cash_rate=0.03):
        """Calculate portfolio returns from start date to end date
        
        Args:
            end_date: End date for calculation (str in 'YYYY-MM-DD' format or datetime)
            cash_rate: Annual cash return rate for cash holdings (default: 3%)
        
        Returns:
            DataFrame with portfolio NAV and return history
        """
        # Convert date strings to datetime if needed
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Get common date range for all instruments
        start_date = max([data.index.min() for name, data in self.instruments.items() if data is not None and not data.empty], default=self.start_date)
        if start_date < self.start_date:
            start_date = self.start_date
            
        if end_date is None:
            end_date = min([data.index.max() for name, data in self.instruments.items() if data is not None and not data.empty], default=None)
            if end_date is None:
                print("No instrument data available. Cannot calculate returns.")
                return None
        
        # Create a complete daily date range
        daily_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Calculate daily cash return (compounded daily)
        daily_cash_return = (1 + cash_rate) ** (1/252) - 1
        
        # Initialize tracking values
        instrument_navs = {name: self.holdings.get(name, 0) for name in self.instruments.keys()}
        cash_nav = self.holdings.get('Cash', 0)
        total_nav = sum(instrument_navs.values()) + cash_nav
        
        nav_records = []
        return_records = []
        
        # Calculate NAV and returns for each day
        prev_total_nav = total_nav
        prev_instrument_navs = instrument_navs.copy()
        prev_cash_nav = cash_nav
        
        for date in daily_range:
            # Update instrument NAVs using synthetic returns
            for name, prev_nav in prev_instrument_navs.items():
                if name in self.instruments and self.instruments[name] is not None and date in self.instruments[name].index:
                    return_rate = self.instruments[name].loc[date, 'Synthetic_Returns']
                    instrument_navs[name] = prev_nav * (1 + return_rate)
                else:
                    instrument_navs[name] = prev_nav  # No change if data not available
            
            # Update cash NAV
            cash_nav = prev_cash_nav * (1 + daily_cash_return)
            
            # Calculate total portfolio NAV (uncapped)
            total_nav = sum(instrument_navs.values()) + cash_nav
            
            # Calculate period returns
            total_return = (total_nav / prev_total_nav) - 1 if prev_total_nav > 0 else 0
            
            # Store values
            nav_record = {'Date': date, 'NAV': total_nav, 'Cash_NAV': cash_nav}
            for name, nav in instrument_navs.items():
                nav_record[f'{name}_NAV'] = nav
            
            nav_records.append(nav_record)
            
            return_records.append({
                'Date': date,
                'Return': total_return
            })
            
            # Update previous values for next iteration
            prev_total_nav = total_nav
            prev_instrument_navs = instrument_navs.copy()
            prev_cash_nav = cash_nav
        
        # Create history DataFrames
        self.nav_history = pd.DataFrame(nav_records).set_index('Date')
        self.returns_history = pd.DataFrame(return_records).set_index('Date')
        
        # Update current capital to latest NAV
        self.current_capital = total_nav
        
        # Calculate simple performance metrics
        annualized_return = ((1 + self.returns_history['Return']).prod()) ** (252/len(self.returns_history)) - 1
        volatility = self.returns_history['Return'].std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(self.nav_history['NAV'])
        
        print(f"\nPortfolio Performance ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}):")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final NAV: ${total_nav:,.2f}")
        print(f"Number of trading days: {len(self.returns_history)}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        
        return self.nav_history
    
    def _calculate_max_drawdown(self, nav_series):
        """Calculate maximum drawdown from NAV series"""
        roll_max = nav_series.cummax()
        drawdown = (nav_series - roll_max) / roll_max
        return drawdown.min()
    
    def plot_performance(self, save_path=None):
        """Plot portfolio performance"""
        if self.nav_history.empty:
            print("No performance data available. Run calculate_returns() first.")
            return
            
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot NAV values
        ax1.plot(self.nav_history.index, self.nav_history['NAV'], 'b-', label='Portfolio NAV')
        for name in self.instruments.keys():
            if f'{name}_NAV' in self.nav_history.columns:
                ax1.plot(self.nav_history.index, self.nav_history[f'{name}_NAV'], '-', label=f'{name} NAV')
                
        if 'Cash_NAV' in self.nav_history.columns:
            ax1.plot(self.nav_history.index, self.nav_history['Cash_NAV'], 'g-', label='Cash NAV')
            
        ax1.set_title('Portfolio Performance')
        ax1.set_ylabel('NAV ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot allocation over time
        allocation_data = {}
        for name in list(self.instruments.keys()) + ['Cash']:
            nav_col = f'{name}_NAV'
            if nav_col in self.nav_history.columns:
                allocation_data[name] = self.nav_history[nav_col] / self.nav_history['NAV'] * 100
        
        if allocation_data:
            allocation_df = pd.DataFrame(allocation_data)
            allocation_df.plot.area(ax=ax2, alpha=0.5)
            ax2.set_ylabel('Allocation (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Performance chart saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, folder_path=None):
        """Save portfolio results to CSV files"""
        if self.nav_history.empty:
            print("No performance data available. Run calculate_returns() first.")
            return
            
        if folder_path is None:
            folder_path = os.getcwd()
            
        # Ensure folder exists
        os.makedirs(folder_path, exist_ok=True)
        
        # Save NAV history
        nav_path = os.path.join(folder_path, 'portfolio_nav_history.csv')
        self.nav_history.to_csv(nav_path)
        
        # Save returns history
        returns_path = os.path.join(folder_path, 'portfolio_returns_history.csv')
        self.returns_history.to_csv(returns_path)
        
        print(f"Portfolio results saved to {folder_path}")

def create_portfolio_with_custom_allocation(instruments_allocation, initial_capital=1000000.0, start_date='2014-01-01', end_date='2024-12-31', cash_rate=0.025):
    """
    Create and calculate a portfolio with custom instrument allocations.
    
    Args:
        instruments_allocation (dict): Dictionary with instrument names as keys and allocation percentages as values.
                                      Example: {'Gold': 30.0, 'Apple': 25.0}
        initial_capital (float): Initial capital for the portfolio.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        cash_rate (float): Annual cash return rate.
        
    Returns:
        Portfolio: Configured portfolio with calculated returns.
    """
    # Validate total allocation
    total_allocation = sum(instruments_allocation.values())
    if abs(total_allocation - 100.0) > 0.001:
        remaining = 100.0 - total_allocation
        print(f"Warning: Allocations sum to {total_allocation}%. Adding {remaining}% to Cash.")
        instruments_allocation['Cash'] = instruments_allocation.get('Cash', 0.0) + remaining
    
    # Create portfolio
    portfolio = Portfolio(initial_capital=initial_capital, start_date=start_date)
    
    # Set allocation
    portfolio.set_allocation(**instruments_allocation)
    
    # Calculate returns
    portfolio.calculate_returns(end_date=end_date, cash_rate=cash_rate)
    
    return portfolio

def plot_portfolio_performance(portfolio, save_prefix):
    """
    Create separate plots for cumulative NAV and individual position NAVs.
    
    Args:
        portfolio: Portfolio object with calculated returns
        save_prefix: Prefix for saving plot files
    """
    if portfolio.nav_history.empty:
        print("No performance data available. Run calculate_returns() first.")
        return
    
    # Create figure for cumulative NAV
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(portfolio.nav_history.index, portfolio.nav_history['NAV'], 'b-', linewidth=2)
    ax1.set_title('Portfolio Cumulative NAV')
    ax1.set_ylabel('NAV ($)')
    ax1.set_xlabel('Date')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_cumulative_nav.png', dpi=300)
    plt.close(fig1)
    
    # Create figure for individual positions
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Plot individual position NAVs
    for name in list(portfolio.instruments.keys()) + ['Cash']:
        nav_col = f'{name}_NAV'
        if nav_col in portfolio.nav_history.columns:
            ax2.plot(portfolio.nav_history.index, portfolio.nav_history[nav_col], '-', label=f'{name}')
    
    ax2.set_title('Individual Position NAVs')
    ax2.set_ylabel('NAV ($)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_individual_navs.png', dpi=300)
    plt.close(fig2)
    
    # Create figure for allocation over time
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    allocation_data = {}
    for name in list(portfolio.instruments.keys()) + ['Cash']:
        nav_col = f'{name}_NAV'
        if nav_col in portfolio.nav_history.columns:
            allocation_data[name] = portfolio.nav_history[nav_col] / portfolio.nav_history['NAV'] * 100
    
    if allocation_data:
        allocation_df = pd.DataFrame(allocation_data)
        allocation_df.plot.area(ax=ax3, alpha=0.5)
        ax3.set_ylabel('Allocation (%)')
        ax3.set_xlabel('Date')
        ax3.set_title('Portfolio Allocation Over Time')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_allocation.png', dpi=300)
    plt.close(fig3)
    
    print(f"Performance charts saved with prefix: {save_prefix}")

def main():
    # Modified allocation with all assets - reducing SPX ATM Call allocation
    # due to high volatility and increasing cash position
    custom_allocation = {
        'Gold': 10.0,
        'Apple': 5.0,
        'JnJ': 5.0,
        'PG': 5.0,
        'EM': 10.0,
        'Private Credit': 10.0,
        'TIPS ETF': 10.0,
        '10Y Bond': 10.0,
        #'SPX ATM Call': 10.0,
        'Convertible Bond' : 10.0,
        'Inflation Swap' :10.0,
        'Cash': 15.0  
    }
    
    # Output directory
    output_dir = 'portfolio_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create portfolio with custom allocation
    portfolio = create_portfolio_with_custom_allocation(custom_allocation)
    
    # Plot performance and save to output directory
    plot_portfolio_performance(portfolio, os.path.join(output_dir, 'portfolio'))
    
    # Save results to output directory
    portfolio.save_results(output_dir)

if __name__ == "__main__":
    main()
