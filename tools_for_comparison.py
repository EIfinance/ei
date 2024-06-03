from Closetoclose.py import close_to_close
from GarmanKlass.py import garman_klass_estimator
from Parkinson.py import Parkinson
from Roger_Satchell.py import roger_satchell_volatility
from Yang_Zhang.py import *


# Correlation between the five volatility estimators

k = 1 # Set the value k

A = close_to_close(data)
B = GarmanKlass__estimator(data)
C = Parkinson(data)
D = roger_satchell_volatility(data)
E = calculate_yang_zhang(data, k)

# A list of the 5 estimators (as dataframes)
estimators = [A,B,C,D,E]

# Computing the correlation matrix
correlation_matrix = np.zeros((5,5))
for i in range(len(estimators)):
    for j in range(len(estimators)):
        correlation_matrix[i,j] = estimators[i].corr(estimators[j])

print(correlation_matrix)


# Correlation with the true volatility

true_volatility = ... 
correlation_with_true_vol = np.zeros((5))
for i in range(len(correlation_with_true_vol)):
    correlation_with_true_vol[i] = estimators[i].corr(true_volatility)


# Efficiency

true_variance = true_volatility.var()

# Computing the efficiency of each estimators compared to the true volatility
efficiency_vector = np.zeros((5))
for i in range(len(efficiency_vector)):
    efficiency_vector[i] = true_variance/(estimators[i].var())
