import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def error_function(m, b, x, y):
    total_error = 0
    n = len(x)
    for i in range(n):
        y_pred = 0
        for j in range(len(m)):
            y_pred += m[j] * x[i][j]
        y_pred += b
        total_error += (y[i] - y_pred) ** 2
    return total_error / len(x)

def model(X, y, learning_rate, iterations):
    n_sample, n_features = X.shape
    
    m = np.zeros(n_features)
    b = 0

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    for _ in range(iterations):
        y_pred = np.dot(X, m) + b

        dm = -(2/n_sample) * np.dot(X.T, (y - y_pred))
        db = -(2/n_sample) * np.sum(y - y_pred)

        m -= learning_rate * dm
        b -= learning_rate * db

        return m, b, mean, std

path = "C:\\Users\\gabri\\.cache\\kagglehub\\datasets\\fratzcan\\usa-house-prices\\versions\\1\\USA Housing Dataset.csv"
df = pd.read_csv(path)
df = df.drop(['date', 'street', 'city', 'statezip', 'country'], axis=1)

print(df.corr()['price'])
df = df.drop(['sqft_lot', 'condition', 'yr_renovated', 'yr_built'], axis=1)
