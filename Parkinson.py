import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = yf.download('SPY', start='2000-01-01', end='2014-01-01')

def Parkinson(data):
  data['Parkinson'] = np.sqrt(1 / (4 * np.log(2)) * np.log(data['High'] / data['Low']) ** 2)
  return data
