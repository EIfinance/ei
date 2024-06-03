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
