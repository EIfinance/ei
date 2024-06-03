import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Parkinson(data):
  data['Parkinson'] = np.sqrt(1 / (4 * np.log(2)) * np.log(data['High'] / data['Low']) ** 2)
  return data
