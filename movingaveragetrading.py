import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download SPY historical data from Yahoo Finance
spy_data = yf.download('SPY', start='2020-01-01', end='2023-01-01')

# Define short-term and long-term moving averages
short_window = 20
long_window = 200

# Calculate the short-term and long-term moving averages
spy_data['short_mavg'] = spy_data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
spy_data['long_mavg'] = spy_data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
spy_data['signal'] = 0.0
spy_data['signal'][short_window:] = np.where(spy_data['short_mavg'][short_window:] > spy_data['long_mavg'][short_window:], 1.0, 0.0)

# Generate trading orders
spy_data['positions'] = spy_data['signal'].diff()

# Initialize the portfolio
initial_capital = 100000.0  # Initial capital in USD

# Create a DataFrame 'positions'
positions = pd.DataFrame(index=spy_data.index).fillna(0.0)
positions['SPY'] = spy_data['signal']  # This implies all in on SPY when the signal is 1

# Initialize the portfolio with value owned
portfolio = positions.multiply(spy_data['Adj Close'], axis=0)

# Store the difference in shares owned
pos_diff = positions.diff()

# Add `holdings` to portfolio
portfolio['holdings'] = (positions.multiply(spy_data['Adj Close'], axis=0)).sum(axis=1)

# Add `cash` to portfolio
portfolio['cash'] = initial_capital - (pos_diff.multiply(spy_data['Adj Close'], axis=0)).sum(axis=1).cumsum()   

# Add `total` to portfolio
portfolio['total'] = portfolio['cash'] + portfolio['holdings']

# Add `returns` to portfolio
portfolio['returns'] = portfolio['total'].pct_change()

# Compute the cumulative returns
portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod() - 1

# Compute the cumulative P&L
portfolio['cumulative_pnl'] = portfolio['total'] - initial_capital

# Calculate the Sharpe Ratio
risk_free_rate = 0
sharpe_ratio = (portfolio['returns'].mean() - risk_free_rate) / portfolio['returns'].std()

# Calculate the maximum drawdown
rolling_max = portfolio['total'].cummax()
daily_drawdown = portfolio['total'] / rolling_max - 1.0
max_drawdown = daily_drawdown.cummin().min()

# Calculate additional risk metrics

# Volatility (standard deviation of returns)
volatility = portfolio['returns'].std()

print('volatility :',volatility)
print('Sharpe ratio :', sharpe_ratio)
print('Max drawdown :', max_drawdown)
# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

# Plot the closing price and the moving averages
ax1.plot(spy_data['Close'], label='SPY Close Price')
ax1.plot(spy_data['short_mavg'], label='40-day MA')
ax1.plot(spy_data['long_mavg'], label='100-day MA')

# Plot buy signals
ax1.plot(spy_data[spy_data['positions'] == 1.0].index, 
         spy_data['short_mavg'][spy_data['positions'] == 1.0], 
         '^', markersize=10, color='g', lw=0, label='Buy Signal')

# Plot sell signals
ax1.plot(spy_data[spy_data['positions'] == -1.0].index, 
         spy_data['short_mavg'][spy_data['positions'] == -1.0], 
         'v', markersize=10, color='r', lw=0, label='Sell Signal')

ax1.set_title('SPY Price and Trading Signals')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()

# Plot the returns
ax2.plot(portfolio['returns'], label='Returns')
ax2.set_title('Strategy Returns')
ax2.set_xlabel('Date')
ax2.set_ylabel('Returns')
ax2.legend()

# Plot the cumulative P&L
ax3.plot(portfolio['cumulative_pnl'], label='Cumulative P&L')
ax3.set_title('Cumulative P&L')
ax3.set_xlabel('Date')
ax3.set_ylabel('Cumulative P&L')
ax3.legend()

plt.tight_layout()
plt.show()
