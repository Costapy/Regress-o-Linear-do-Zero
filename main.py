import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def error_function(m, b, X, y):
    y_pred = np.dot(X, m) + b
    mse = np.mean((y - y_pred) ** 2)
    return mse

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
    
def predict(X, m, b, mean, std):
    X = (X - mean) / std
    return np.dot(X, m) + b

path = "C:\\Users\\gabri\\.cache\\kagglehub\\datasets\\fratzcan\\usa-house-prices\\versions\\1\\USA Housing Dataset.csv"
df = pd.read_csv(path)
df = df.drop(['date', 'street', 'city', 'statezip', 'country'], axis=1)

print(df.corr()['price'])
df = df.drop(['sqft_lot', 'condition', 'yr_renovated', 'yr_built'], axis=1)

test_size = 0.2

df_test = df.sample(frac=test_size, random_state=42)
df_train = df.drop(df_test.index)

X = df_train.drop('price', axis=1).values
y = df_train['price'].values

learning_rate = 0.01
iterations = 1000
m, b, mean, std = model(X, y, learning_rate, iterations)

X_test = df_test.drop('price', axis=1).values
y_test = df_test['price'].values

y_pred = predict(X_test, m, b, mean, std)

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices') 
plt.title('Actual vs Predicted Prices')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)
plt.show()