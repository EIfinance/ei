import numpy as np
import pandas as pd
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data = yf.download('SPY', start='2000-01-01', end='2014-01-01')
def garman_klass__estimator(data):
    sigma_GK = 0.5 * (np.log(data['High']/data['Low']))**2 - (2*np.log(2)-1) * (np.log(data['Close']/data['Open']))**2
    sigma_GK = np.sqrt(sigma_GK)
    return sigma_GK