import pandas as pd
import matplotlib.pyplot as plt

def error_function(m, b, x, y):
    #error = y - y_pred
    total_error = 0
    n = len(x)
    for i in range(n):
        y_pred = 0
        for j in range(len(m)):
            y_pred += m[j] * x[i][j]
        y_pred += b
        total_error += (y[i] - y_pred) ** 2
    return total_error / len(x)


path = "C:\\Users\\gabri\\.cache\\kagglehub\\datasets\\fratzcan\\usa-house-prices\\versions\\1\\USA Housing Dataset.csv"
df = pd.read_csv(path)

x = df.drop("price", axis=1).values
y = df["price"].values
print(x)
print(y)
