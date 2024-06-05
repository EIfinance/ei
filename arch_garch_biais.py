import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt

def download_data(ticker='SPY', start='2000-01-01', end='2013-12-31'):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Open', 'High', 'Low', 'Close']]
    return data

# GARCH
def apply_garch_model(log_returns, p, q):
    p, q = int(p), int(q)
    model = arch_model(log_returns, vol='Garch', p=p, q=q, rescale=False)
    model_fit = model.fit(disp="off")
    forecasts = model_fit.forecast(horizon=1, start=0)
    predicted_vol = np.sqrt(forecasts.variance.values.flatten())
    return predicted_vol

# ARCH
def apply_arch_model(log_returns, p):
    p = int(p)
    model = arch_model(log_returns, vol='ARCH', p=p, rescale=False)
    model_fit = model.fit(disp="off")
    forecasts = model_fit.forecast(horizon=1, start=0)
    predicted_vol = np.sqrt(forecasts.variance.values.flatten())
    return predicted_vol

# variances hebdomadaires
def calculate_weekly_variance(volatility, index):
    volatility_series = pd.Series(volatility, index=index)
    var_weekly = volatility_series.resample('W').var()
    return var_weekly

# biais hebdomadaire
def calculate_weekly_bias(realized_vol, predicted_vol, index):
    var_realized_weekly = calculate_weekly_variance(realized_vol, index)
    var_predicted_weekly = calculate_weekly_variance(predicted_vol, index)
    bias_weekly = var_realized_weekly - var_predicted_weekly
    return bias_weekly

if __name__ == "__main__":
    data = download_data()
    
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    
    realized_vol = log_returns.abs()
  
    results = []

    # GARCH pour p, q dans [1, 2, 3]
    for p in range(1, 4):
        for q in range(1, 4):
            predicted_vol_garch = apply_garch_model(log_returns, p, q)
            predicted_vol_garch = predicted_vol_garch[-len(realized_vol):]
            bias_weekly_garch = calculate_weekly_bias(realized_vol, predicted_vol_garch, realized_vol.index)
            avg_bias_garch = bias_weekly_garch.mean()
            results.append(('GARCH', p, q, avg_bias_garch))

    # ARCH pour p dans [1, 2, 3]
    for p in range(1, 4):
        predicted_vol_arch = apply_arch_model(log_returns, p)
        predicted_vol_arch = predicted_vol_arch[-len(realized_vol):]
        bias_weekly_arch = calculate_weekly_bias(realized_vol, predicted_vol_arch, realized_vol.index)
        avg_bias_arch = bias_weekly_arch.mean()
        results.append(('ARCH', p, 0, avg_bias_arch))

    results_df = pd.DataFrame(results, columns=['Model', 'p', 'q', 'Avg_Weekly_Bias'])
    
    print(results_df)

    num_models = len(results_df)
    num_cols = 4
    num_rows = (num_models + num_cols - 1) // num_cols 

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 6))
    fig.suptitle('Histogramme des biais hebdomadaires pour les modèles GARCH et ARCH')

    for index, (model, p, q, avg_bias) in enumerate(results):
        if model == 'GARCH':
            predicted_vol = apply_garch_model(log_returns, p, q)
        else:
            predicted_vol = apply_arch_model(log_returns, p)
        
        predicted_vol = predicted_vol[-len(realized_vol):]
        bias_weekly = calculate_weekly_bias(realized_vol, predicted_vol, realized_vol.index)
        
        ax = axs[index // num_cols, index % num_cols]
        ax.hist(bias_weekly.dropna(), bins=50, edgecolor='black')
        ax.set_title(f'{model} (p={p}, q={q})')
        ax.set_xlabel('Biais')
        ax.set_ylabel('Fréquence')
        ax.text(0.95, 0.95, f'Avg Bias: {avg_bias:.6f}', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    for i in range(num_models, num_rows * num_cols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
