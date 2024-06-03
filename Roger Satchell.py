import numpy as np
import pandas as pd
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data = yf.download('SPY', start='2000-01-01', end='2014-01-01')
def rogers_satchell_volatility(data):
    high = data['High']
    low = data['Low']
    close = data['Close']
    open = data['Open']

    log_hc = np.log(high / close)
    log_ho = np.log(high / open)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open)

    sigma_rs = np.sqrt(log_hc * log_ho - log_lc * log_lo)
    return sigma_rs