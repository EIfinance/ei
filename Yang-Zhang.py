import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_yang_zhang(data, k):
    """
    Calcule l'indicateur Yang-Zhang pour une valeur donnée de k.
    """
    # Calculer l'indicateur Yang-Zhang
    yang_zhang = np.sqrt(
        np.log(data['Open'] / data['Close'].shift(1)) ** 2
        + k * np.log(data['Open'] / data['Close']) ** 2
        + (1 - k) * (np.log(data['High'] / data['Close']) * np.log(data['High'] / data['Open'])
                  + np.log(data['Low'] / data['Close']) * np.log(data['Low'] / data['Open'])))
    
    return yang_zhang

def plot_yang_zhang_variance(data):
    """
    Trace la variance de l'indicateur Yang-Zhang en fonction de k.
    """
    k_values = np.linspace(0, 1, 100)
    variances = []

    for k in k_values:
        yang_zhang = calculate_yang_zhang(data, k)
        variances.append(np.var(yang_zhang))

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, variances, label='Variance de Yang-Zhang')
    plt.xlabel('k')
    plt.ylabel('Variance')
    plt.title('Variance de l\'indicateur Yang-Zhang en fonction de k')
    plt.legend()
    plt.grid(True)
    plt.show()
