import numpy as np
import pandas as pd
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data = yf.download('SPY', start='2000-01-01', end='2014-01-01')
def close_to_close(data):
    data['Close-to-close'] = np.abs(np.log(data['Close'] / data['Close'].shift(1)))
    Close_to_close=data['Close-to-close']
    return Close_to_close


