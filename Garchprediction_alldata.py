import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Load data
data = pd.read_csv('RealizedVarianceData.csv', low_memory=False)

# Convert dates and handle invalid values
data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'], errors='coerce')
data = data.dropna(subset=['Unnamed: 0', 'Realized Variance (5-minute)'])

# Convert 'Realized Variance (5-minute)' to numeric and handle errors
data['Realized Variance (5-minute)'] = pd.to_numeric(data['Realized Variance (5-minute)'], errors='coerce')
data = data.dropna(subset=['Realized Variance (5-minute)'])

# Rename the date column
data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
data.set_index('Date', inplace=True)

# Use the realized variance directly
data['Realized Volatility (5-minute)'] = np.sqrt(data['Realized Variance (5-minute)']*100)

# Split data into in-sample and out-sample sets
split_date = '2010-01-01'
in_sample = data.loc[:split_date]
out_sample = data.loc[split_date:]

a=1
b=0
# Rolling window forecast
window_size = 30  
forecasts = []

for start in range(len(out_sample) - window_size):
    end = start + window_size
    train_data = pd.concat([in_sample['Realized Volatility (5-minute)'], out_sample['Realized Volatility (5-minute)'][:start]])
    
    # Fit the GARCH model
    model = arch_model(train_data, vol='Garch', p=a, q=b)
    model_fit = model.fit(disp='off')
    if b==0:    
       model_fit.params['beta[1]']=0
    # Manually forecast the volatility
    last_volatility = train_data.iloc[-1]  # Last observed volatility
    forecast_variance = model_fit.params['omega'] + model_fit.params['alpha[1]'] * (last_volatility ** 2) + model_fit.params['beta[1]'] * model_fit.conditional_volatility[-1] ** 2
    forecast_volatility = np.sqrt(forecast_variance)
    forecasts.append(forecast_volatility)

# Convert forecasts to a Series with correct index
forecasts = pd.Series(forecasts, index=out_sample.index[window_size:])

# Plot the true out-sample data and the forecasted data
plt.figure(figsize=(12, 6))
plt.plot(out_sample.index[window_size:], out_sample['Realized Volatility (5-minute)'][window_size:], label='True Out-Sample Volatility')
plt.plot(forecasts.index, forecasts, label='Forecasted Volatility')
plt.legend()
plt.title('True vs Forecasted Volatility')
plt.show()
