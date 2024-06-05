import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

H = 0.14

# Read the data using pandas
sigma = pd.read_csv("RealizedVarianceData.csv")["Realized Variance (5-minute)"].tolist()[1:]

# Backward fill missing values
sigma = pd.Series(sigma).bfill()

# Convert each element to float and take the logarithm
sigma = np.array([np.log10(float(x)) for x in sigma])

covariances = []

for lag in range(0,50):
    sigma_translated = np.roll(sigma, -lag)
    sigma_translated[-lag:] = np.nan
    covariance = np.ma.cov(np.ma.array(sigma, mask=np.isnan(sigma_translated)),
                           np.ma.array(sigma_translated, mask=np.isnan(sigma_translated)))[0, 1]
    covariances.append(covariance)

coeff=np.polyfit([k**(2*H) for k in range (1,50)], covariances[1:], 1)
x=[k**(2*H) for k in range (1,50)]
y=[(coeff[1]+coeff[0]*k) for k in x]

plt.figure(figsize=(10, 5))
plt.plot([k**(2*H) for k in range (50)], covariances)
plt.xlabel('Lag puissance 2H')
plt.ylabel('Covariance des log sigma')
plt.grid(True)
plt.plot(x,y)
plt.show()
