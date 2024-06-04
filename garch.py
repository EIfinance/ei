import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# Lire les données CSV avec l'option low_memory=False
data = pd.read_csv('/Users/nicolas/Desktop/EI/RealizedVarianceData.csv', low_memory=False)

# Convertir les dates en datetime en gérant les valeurs non valides
data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'], errors='coerce')

# Filtrer les valeurs manquantes et non valides
data = data.dropna(subset=['Unnamed: 0', 'Realized Variance (5-minute)'])

# Convertir les valeurs de la colonne 'Realized Variance (5-minute)' en numérique
data['Realized Variance (5-minute)'] = pd.to_numeric(data['Realized Variance (5-minute)'], errors='coerce')

# Filtrer à nouveau les valeurs non valides après conversion en numérique
data = data.dropna(subset=['Realized Variance (5-minute)'])

# Renommer la colonne 'Unnamed: 0' en 'Date'
data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# Définir la colonne 'Date' comme index
data.set_index('Date', inplace=True)

# Convertir la variance réalisée en volatilité (en prenant la racine carrée)
data['Realized Volatility (5-minute)'] = np.sqrt(data['Realized Variance (5-minute)'])

# Séparer les données d'entraînement et de test
train_data = data.loc['2000-01-01':'2010-12-31']
test_data = data.loc['2011-01-01':'2013-12-31']


# Extraire les volatilités réalisées pour l'entraînement et le test
train_volatility = train_data['Realized Volatility (5-minute)']
test_volatility = test_data['Realized Volatility (5-minute)']

# Rescaler les volatilités pour l'entraînement
returns = 100 * train_volatility

rolling_predictions = []
rolling_predictions_2 = []
test_size = 365

for i in range(test_size):
    train = returns[:-(test_size-i)]
    model = arch_model(train, p=3, q=0)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
    model_2 = arch_model(train, vol='Garch', p=2, q=1)
    model_fit_2 = model.fit(disp='off')
    pred_2 = model_fit_2.forecast(horizon=1)
    rolling_predictions_2.append(np.sqrt(pred_2.variance.values[-1,:][0]))

rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365:])
rolling_predictions_2 = pd.Series(rolling_predictions_2, index=returns.index[-365:])

plt.figure(figsize=(10,4))
true, = plt.plot(returns[-365:])
plt.plot(rolling_predictions, color='red')
plt.plot(rolling_predictions_2, color='green')
plt.show()
